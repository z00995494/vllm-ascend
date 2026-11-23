# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.ops import agmm
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.prepared_linear_input import PreparedLinearInput


def make_layer(tp_size=2):
    scheme = AscendW8A8DynamicLinearMethod()
    adapter = AscendLinearMethod.__new__(AscendLinearMethod)
    adapter.quant_method = scheme
    return SimpleNamespace(
        custom_op=None,
        gather_output=False,
        bias=None,
        tp_size=tp_size,
        tp_rank=0,
        quant_method=adapter,
        weight=torch.ones(256, 32, dtype=torch.int8),
        weight_scale_fp32=torch.full((32,), 0.25),
        skip_bias_add=False,
        return_bias=True,
        input_size_per_partition=256,
        output_size=32 * tp_size,
        prefix="model.layers.0.mlp.gate_up_proj",
    )


def make_config(**overrides):
    config = SimpleNamespace(
        lora_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_op(layer, min_tokens=1, max_tokens=65535):
    return agmm.SequenceParallelAllGatherMMOp(layer, min_tokens=min_tokens, max_tokens=max_tokens)


@pytest.fixture
def harness():
    layer = make_layer()
    group = Mock(world_size=2, rank_in_group=0)
    group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "original-tp"
    metadata = SimpleNamespace(num_prefills=1, num_decodes=0)
    context = SimpleNamespace(
        attn_metadata={"model.layers.0.self_attn.attn": metadata},
        no_compile_layers={},
    )
    with (
        patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=group),
        patch.object(agmm, "is_forward_context_available", return_value=True),
        patch.object(agmm, "get_forward_context", return_value=context),
        patch.object(agmm.torch_npu, "get_npu_format", return_value=2) as weight_format,
        patch.object(agmm.torch_npu, "npu_all_gather_quant_mm", create=True) as fused,
        patch.object(agmm, "sp_all_gather") as all_gather,
    ):
        op = make_op(layer)

        def prepare(owner, x):
            return PreparedLinearInput(x.to(torch.int8), torch.full((x.shape[0],), 0.5), x.dtype)

        def gather(x):
            return x.repeat(group.world_size, 1)

        def projection(owner, x, bias):
            return (x.to(torch.int8).float().matmul(owner.weight.float()) * 0.5 * owner.weight_scale_fp32).to(x.dtype)

        def fused_projection(x, weight, hcom, world_size, **kwargs):
            gathered = x.repeat(world_size, 1)
            output = (
                gathered.float().matmul(weight.float()) * kwargs["x1_scale"].repeat(world_size, 1) * kwargs["x2_scale"]
            ).to(kwargs["y_dtype"])
            return output, gathered, torch.empty(0)

        layer.quant_method.quant_method.prepare_input_for_overlap = Mock(side_effect=prepare)
        layer.quant_method.apply = Mock(side_effect=projection)
        all_gather.side_effect = gather
        fused.side_effect = fused_projection
        yield SimpleNamespace(
            layer=layer,
            op=op,
            group=group,
            metadata=metadata,
            context=context,
            fused=fused,
            all_gather=all_gather,
            gather=gather,
            projection=projection,
            weight_format=weight_format,
        )


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("tokens", [64, 65, 72])
def test_fused_projection_matches_all_gather_plus_mm(harness, tp_size, tokens):
    h = harness
    h.group.world_size = tp_size
    x = torch.arange(tokens * 256).remainder(4).reshape(tokens, 256).to(torch.bfloat16)
    for rank in range(tp_size):
        h.group.rank_in_group = rank
        expected = h.projection(h.layer, h.gather(x), None)
        actual = h.op.run(x)
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (tokens * tp_size, 32)
    h.all_gather.assert_not_called()
    h.layer.quant_method.apply.assert_not_called()
    args, kwargs = h.fused.call_args
    assert args[0].shape == (tokens, 256)  # Quantize only the rank-local shard.
    assert args[1] is h.layer.weight  # Loaded INT8 weights are already [K, N].
    assert args[2:4] == ("original-tp", tp_size)
    assert kwargs["x1_scale"].shape == (tokens, 1)
    assert kwargs["x2_scale"].shape == (1, 32)
    assert kwargs["comm_mode"] == "aiv"
    assert kwargs["y_dtype"] == torch.bfloat16
    assert kwargs["gather_output"] is True


@pytest.mark.parametrize("rows,expected", [(8191, False), (8192, True), (20480, True), (20481, False)])
def test_token_limit_boundaries_use_the_full_sequence(harness, rows, expected):
    # The token gate sees the full sequence (shard rows x TP size); the INT8
    # benefit range is 16k-40k inclusive (exclusive upper bound).
    h = harness
    op = make_op(h.layer, min_tokens=16384, max_tokens=40961)
    x = torch.ones(rows, 256, dtype=torch.bfloat16)
    assert op._can_fuse(x) is expected
    torch.testing.assert_close(op.run(x), h.projection(h.layer, h.gather(x), None))
    assert h.fused.called is expected
    assert h.all_gather.called is not expected


@pytest.mark.parametrize("prefills,decodes", [(0, 2), (1, 1)])
def test_decode_and_mixed_batches_keep_all_gather_plus_mm(harness, prefills, decodes):
    h = harness
    h.metadata.num_prefills = prefills
    h.metadata.num_decodes = decodes
    x = torch.ones(64, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(h.op.run(x), h.projection(h.layer, h.gather(x), None))
    h.fused.assert_not_called()
    h.all_gather.assert_called_once_with(x)
    h.layer.quant_method.apply.assert_called_once()


@pytest.mark.parametrize("metadata", [None, {}, SimpleNamespace(num_prefills=0, num_decodes=1)])
def test_missing_metadata_falls_back(harness, metadata):
    h = harness
    h.context.attn_metadata = metadata
    x = torch.ones(64, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(h.op.run(x), h.projection(h.layer, h.gather(x), None))
    h.fused.assert_not_called()
    h.all_gather.assert_called_once_with(x)


def test_any_attention_metadata_entry_describes_the_batch(harness):
    # The MLP owns no attention metadata; any entry carries the same
    # prefill/decode split.
    h = harness
    h.context.attn_metadata = {"unrelated.attn": SimpleNamespace(num_prefills=1, num_decodes=0)}
    x = torch.ones(64, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(h.op.run(x), h.projection(h.layer, h.gather(x), None))
    h.fused.assert_called_once()
    h.all_gather.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_runtime_fallback_preserves_all_gather_plus_mm(harness, dtype):
    h = harness
    x = torch.ones(3, 256, dtype=dtype)
    expected = h.projection(h.layer, h.gather(x), None)
    torch.testing.assert_close(h.op.run(x), expected)
    h.all_gather.assert_called_once_with(x)
    h.layer.quant_method.apply.assert_called_once()
    h.fused.assert_not_called()


@pytest.mark.parametrize(
    "bad_case",
    [
        "weight_dtype",
        "weight_shape",
        "noncontiguous_weight",
        "missing_scale",
        "scale_dtype",
        "scale_shape",
        "format",
        "small_k",
        "large_k",
        "small_m",
        "empty",
        "rank",
    ],
)
def test_unsupported_runtime_layout_is_rejected_before_collective(harness, bad_case):
    h = harness
    x = torch.ones(64, 256, dtype=torch.bfloat16)
    if bad_case == "weight_dtype":
        h.layer.weight = h.layer.weight.float()
    elif bad_case == "weight_shape":
        h.layer.weight = h.layer.weight.t().contiguous()
    elif bad_case == "noncontiguous_weight":
        h.layer.weight = torch.ones(32, 256, dtype=torch.int8).t()
    elif bad_case == "missing_scale":
        h.layer.weight_scale_fp32 = None
    elif bad_case == "scale_dtype":
        h.layer.weight_scale_fp32 = h.layer.weight_scale_fp32.to(torch.bfloat16)
    elif bad_case == "scale_shape":
        h.layer.weight_scale_fp32 = h.layer.weight_scale_fp32.reshape(-1, 1)
    elif bad_case == "format":
        h.weight_format.return_value = 3
    elif bad_case == "small_k":
        x = x[:, :255]
    elif bad_case == "large_k":
        x = torch.ones(64, 65535, dtype=torch.bfloat16)
    elif bad_case == "small_m":
        x = x[:63]
    elif bad_case == "empty":
        x = x[:0]
    elif bad_case == "rank":
        x = x.unsqueeze(0)
    assert not h.op._can_fuse(x)
    h.fused.assert_not_called()


def test_supported_shape_boundaries(harness):
    h = harness
    assert not h.op._can_fuse(torch.ones(63, 256, dtype=torch.bfloat16))
    assert h.op._can_fuse(torch.ones(64, 256, dtype=torch.bfloat16))
    h.layer.weight = torch.ones(65534, 32, dtype=torch.int8)
    assert h.op._can_fuse(torch.ones(64, 65534, dtype=torch.bfloat16))


@pytest.mark.parametrize("weight_format", [2, 29])
def test_nd_and_nz_weights_feed_the_fused_op_as_nd(harness, weight_format):
    h = harness
    h.weight_format.return_value = weight_format
    h.layer.weight = torch.full((256, 32), 2, dtype=torch.int8)
    x = torch.ones(64, 256, dtype=torch.bfloat16)
    with patch.object(agmm.torch_npu, "npu_format_cast", side_effect=lambda w, fmt: w.clone()) as format_cast:
        expected = h.projection(h.layer, h.gather(x), None)
        torch.testing.assert_close(h.op.run(x), expected)
        fed = h.fused.call_args.args[1]
        if weight_format == agmm.ACL_FORMAT_FRACTAL_ND:
            assert fed is h.layer.weight
            format_cast.assert_not_called()
            return
        assert fed is not h.layer.weight
        format_cast.assert_called_once_with(h.layer.weight, agmm.ACL_FORMAT_FRACTAL_ND)
        h.op.run(x)
        assert format_cast.call_count == 1
        h.layer.weight = torch.full((256, 32), 2, dtype=torch.int8)
        h.op.run(x)
        assert format_cast.call_count == 2


def test_static_weight_checks_are_cached_until_weights_change(harness):
    h = harness
    x = torch.ones(64, 256, dtype=torch.bfloat16)
    h.op.run(x)
    assert h.weight_format.call_count == 1
    x2_scale_first = h.op._x2_scale
    h.op.run(x)
    assert h.weight_format.call_count == 1
    assert h.fused.call_args.kwargs["x2_scale"] is x2_scale_first

    # Replacement must not compare two multi-element tensors for equality.
    h.layer.weight = torch.full((256, 32), 2, dtype=torch.int8)
    h.layer.weight_scale_fp32 = torch.full((32,), 0.5)
    h.op.run(x)
    assert h.weight_format.call_count == 2
    assert h.op._x2_scale is not x2_scale_first
    assert (h.fused.call_args.kwargs["x2_scale"] == 0.5).all()

    h.layer.weight.fill_(3)
    h.op.run(x)
    assert h.weight_format.call_count == 3
    h.layer.weight_scale_fp32.fill_(0.75)
    h.op.run(x)
    assert h.weight_format.call_count == 4
    assert (h.fused.call_args.kwargs["x2_scale"] == 0.75).all()


def test_fused_failure_is_not_retried_as_another_collective(harness):
    h = harness
    h.fused.side_effect = RuntimeError("collective failed")
    with pytest.raises(RuntimeError, match="collective failed"):
        h.op.run(torch.ones(64, 256, dtype=torch.bfloat16))
    h.all_gather.assert_not_called()


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("custom_op", object()),
        ("gather_output", True),
        ("bias", torch.ones(32)),
        ("tp_size", 1),
        ("tp_size", 16),
        ("quant_method", object()),
    ],
)
def test_static_rejections_do_not_modify_layer(attribute, value):
    layer = make_layer()
    setattr(layer, attribute, value)
    original = layer.custom_op
    assert agmm.SequenceParallelAllGatherMMOp.unsupported_reason(layer) is not None
    assert layer.custom_op is original


def test_other_quantization_subclasses_are_not_treated_as_int8():
    class OtherScheme(AscendW8A8DynamicLinearMethod):
        pass

    layer = make_layer()
    layer.quant_method.quant_method = OtherScheme()
    assert agmm.SequenceParallelAllGatherMMOp.unsupported_reason(layer) is not None


@pytest.mark.parametrize(
    "gate",
    ["enabled", "no_sp", "lora", "pcp", "dcp", "decode_node", "conflict"],
)
def test_installation_respects_static_gates(harness, gate):
    h = harness
    config = make_config(
        lora_config=object() if gate == "lora" else None,
        kv_transfer_config=(
            SimpleNamespace(is_kv_consumer=True, is_kv_producer=False) if gate == "decode_node" else None
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2 if gate == "pcp" else 1,
            decode_context_parallel_size=2 if gate == "dcp" else 1,
        ),
    )
    if gate == "conflict":
        h.layer.custom_op = object()
    old_op = h.layer.custom_op
    with patch.object(agmm, "get_current_vllm_config", return_value=config):
        enabled = agmm.install_sequence_parallel_agmm(
            h.layer,
            sequence_parallel=gate != "no_sp",
            min_tokens=16384,
            max_tokens=40961,
        )
    assert enabled == (gate == "enabled")
    if enabled:
        assert config.compilation_config.static_forward_context[h.layer.prefix] is h.layer
        assert isinstance(h.layer.custom_op, agmm.SequenceParallelAllGatherMMOp)
        assert h.layer.custom_op.min_tokens == 16384
        assert h.layer.custom_op.max_tokens == 40961
    else:
        assert h.layer.custom_op is old_op
        assert not config.compilation_config.static_forward_context


def test_installation_rejects_duplicate_layer_names(harness):
    h = harness
    other = make_layer()
    config = make_config()
    config.compilation_config.static_forward_context[h.layer.prefix] = other
    with (
        patch.object(agmm, "get_current_vllm_config", return_value=config),
        pytest.raises(ValueError, match="Duplicate layer name"),
    ):
        agmm.install_sequence_parallel_agmm(h.layer, sequence_parallel=True, min_tokens=1, max_tokens=65535)
    assert h.layer.custom_op is None


def test_runtime_dispatch_uses_current_layer(harness):
    h = harness
    h.layer.custom_op = h.op
    h.context.no_compile_layers = {h.layer.prefix: h.layer}
    x = torch.ones(64, 256, dtype=torch.bfloat16)
    actual = agmm._sequence_parallel_agmm(x, h.layer.prefix, 32, 2)
    torch.testing.assert_close(actual, h.projection(h.layer, h.gather(x), None))


def test_fake_registration_returns_full_sequence_without_communication():
    with FakeTensorMode():
        x = torch.empty(3, 256, dtype=torch.bfloat16)
        output = torch.ops.vllm.ascend_sequence_parallel_agmm(x, "unregistered.layer", 32, 2)
        assert output.shape == (6, 32)
        assert output.dtype == x.dtype


def test_linear_tuple_and_tensor_return_contract(harness):
    h = harness
    expected = torch.ones(4, 32, dtype=torch.bfloat16)
    with patch.object(torch.ops.vllm, "ascend_sequence_parallel_agmm", return_value=expected):
        output, bias = h.op.apply(torch.ones(64, 256, dtype=torch.bfloat16))
        assert output is expected and bias is None
        h.op.return_bias = False
        assert h.op.apply(torch.ones(64, 256, dtype=torch.bfloat16)) is expected


def test_run_delegates_to_run_fused_once_gate_passes(harness):
    h = harness
    x = torch.ones(64, 256, dtype=torch.bfloat16)
    with patch.object(h.op, "run_fused", return_value="fused-result") as fused_run:
        assert h.op.run(x) == "fused-result"
        fused_run.assert_called_once_with(x)


def test_aiv_regression_requires_gather_output(harness):
    h = harness
    h.op.run(torch.ones(64, 256, dtype=torch.bfloat16))
    assert h.fused.call_args.kwargs["gather_output"] is True
