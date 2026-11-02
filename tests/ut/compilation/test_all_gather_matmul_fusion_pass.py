# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.compilation.passes import base_pattern
from vllm_ascend.compilation.passes.all_gather_matmul_fusion_pass import (
    AllGatherMatmulFusionPass,
    AllGatherMatmulPattern,
    SlicedAllGatherMatmulPattern,
)
from vllm_ascend.device.device_op import BaseDeviceAdaptor

MODULE = "vllm_ascend.compilation.passes.all_gather_matmul_fusion_pass"
TP_SIZE = 2
GROUP = "tp:agmm"
HCOM = "agmm_hcom"
K = 512
N = 256
GATHER = torch.ops.vllm.all_gather.default
GEMM = torch.ops.vllm.unquantized_gemm.default
SLICE = torch.ops.aten.slice.Tensor


@torch.library.custom_op("agmm_test::fused", mutates_args=())
def _fused(
    x: torch.Tensor, weight: torch.Tensor, hcom: str, world_size: int, *, gather_output: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    # Model distinct input shards without starting a distributed process group.
    gathered = torch.cat([x + rank for rank in range(world_size)])
    return gathered @ weight, gathered if gather_output else x.new_empty(0)


@_fused.register_fake
def _fused_fake(
    x: torch.Tensor, weight: torch.Tensor, hcom: str, world_size: int, *, gather_output: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty((x.shape[0] * world_size, weight.shape[1])), x.new_empty(0)


FUSED = torch.ops.agmm_test.fused.default


def _graph(
    *,
    dtype=torch.bfloat16,
    tp=TP_SIZE,
    group=GROUP,
    dim=0,
    k=K,
    bias=False,
    tokens=None,
    extra_gather_user=False,
    extra_slice_user=False,
    noncontiguous=False,
    weight_dtype=None,
    weight_noncontiguous=False,
    slice_dim=0,
    start=0,
    step=1,
):
    graph = fx.Graph()
    fake = FakeTensorMode()

    def placeholder(name, shape, dt=dtype, transpose=False):
        node = graph.placeholder(name)
        with fake:
            value = torch.empty(shape[::-1] if transpose else shape, dtype=dt)
            node.meta["val"] = value.t() if transpose else value
        return node

    def call(target, args, shape):
        node = graph.call_function(target, args)
        with fake:
            node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    x = placeholder("x", (8, k), transpose=noncontiguous)
    weight = placeholder("weight", (N, k), weight_dtype or dtype, weight_noncontiguous)
    bias_node = placeholder("bias", (N,)) if bias else None
    gathered = call(GATHER, (x, dim, tp, group), (8 * tp, k))
    value = gathered
    rows = 8 * tp
    if tokens is not None:
        slice_args = (gathered, slice_dim, start, tokens) if step == 1 else (gathered, slice_dim, start, tokens, step)
        value = call(SLICE, slice_args, (max(0, min(tokens, rows)), k))
        rows = max(0, min(tokens, rows))
    gemm_args = (value, weight, bias_node) if bias else (value, weight)
    output = call(GEMM, gemm_args, (rows, N))
    outputs = [output]
    if extra_gather_user:
        outputs.append(gathered)
    if extra_slice_user:
        outputs.append(value)
    graph.output(tuple(outputs))
    return fx.GraphModule(torch.nn.Module(), graph)


def _targets(gm):
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


class _Reference(fx.Interpreter):
    def call_function(self, target, args, kwargs):
        if target is GATHER:
            return torch.cat([args[0] + rank for rank in range(args[2])])
        if target is GEMM:
            return torch.nn.functional.linear(*args)
        return super().call_function(target, args, kwargs)


@patch(f"{MODULE}.DeviceOperator.npu_all_gather_base_mm", new=_fused)
class TestAllGatherMatmulFusionPass(TestBase):
    def setUp(self):
        super().setUp()
        # Backend registrations are global; each test owns its registry.
        self.enterContext(patch.object(base_pattern, "_registered_patterns", set()))
        self.enterContext(patch.object(base_pattern.nge, "register_replacement"))

    def _make_pass(self, tp=TP_SIZE, group=GROUP):
        tp_group = MagicMock(unique_name=group)
        tp_group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = HCOM
        config = VllmConfig()
        config.model_config = MagicMock(dtype=torch.bfloat16)
        with (
            patch(f"{MODULE}.get_tp_group", return_value=tp_group),
            patch(f"{MODULE}.get_tensor_model_parallel_world_size", return_value=tp),
        ):
            return AllGatherMatmulFusionPass(config)

    def _apply(self, gm, tp=TP_SIZE):
        fusion = self._make_pass(tp)
        fusion(gm.graph)
        gm.graph.lint()
        gm.recompile()
        return fusion

    def test_direct_and_sliced_patterns(self):
        for dtype in (torch.float16, torch.bfloat16):
            for tp in (2, 4, 8):
                for tokens in (None, 1, 7, 13):
                    with self.subTest(dtype=dtype, tp=tp, tokens=tokens):
                        gm = _graph(dtype=dtype, tp=tp, tokens=tokens)
                        expected = next(n.meta["val"].shape for n in gm.graph.nodes if n.target is GEMM)
                        fusion = self._apply(gm, tp)
                        self.assertEqual(fusion.matched_count, 1)
                        self.assertEqual(_targets(gm).count(FUSED), 1)
                        self.assertNotIn(GATHER, _targets(gm))
                        self.assertNotIn(GEMM, _targets(gm))
                        fused = next(n for n in gm.graph.nodes if n.target is FUSED)
                        self.assertEqual(fused.args[2:4], (HCOM, tp))
                        self.assertFalse(fused.kwargs.get("gather_output", False))
                        self.assertEqual(fused.args[1].args[1], [1, 0])
                        output = next(n for n in gm.graph.nodes if n.op == "output").args[0][0]
                        self.assertEqual(output.meta["val"].shape, expected)

    def test_traced_graph_matches(self):
        config = VllmConfig()
        config.model_config = MagicMock(dtype=torch.bfloat16)
        for pattern_cls in (AllGatherMatmulPattern, SlicedAllGatherMatmulPattern):
            with self.subTest(pattern=pattern_cls.__name__):
                pattern = pattern_cls(config, TP_SIZE, GROUP, HCOM)
                args = [torch.empty(8, K, dtype=torch.bfloat16), torch.empty(N, K, dtype=torch.bfloat16)]
                if pattern_cls is SlicedAllGatherMatmulPattern:
                    args.append(11)
                gm = make_fx(pattern.get_pattern(), tracing_mode="fake")(*args)
                self.assertEqual(self._apply(gm).matched_count, 1)
                self.assertNotIn(GATHER, _targets(gm))
                self.assertNotIn(GEMM, _targets(gm))

    def test_repeated_registration_on_same_matcher(self):
        config = VllmConfig()
        config.model_config = MagicMock(dtype=torch.bfloat16)
        pattern = AllGatherMatmulPattern(config, TP_SIZE, GROUP, HCOM)
        matcher = pm.PatternMatcherPass()
        pattern.register(matcher)
        pattern.register(matcher)
        self.assertEqual(base_pattern.nge.register_replacement.call_count, 1)
        self.assertEqual(matcher.apply(_graph().graph), 1)

    def test_rewrite_preserves_values_on_cpu(self):
        torch.manual_seed(17)
        for dtype in (torch.float16, torch.bfloat16):
            for tokens in (None, 1, 13):
                with self.subTest(dtype=dtype, tokens=tokens):
                    gm = _graph(dtype=dtype, tokens=tokens)
                    x = torch.randn(8, K, dtype=dtype)
                    weight = torch.randn(N, K, dtype=dtype)
                    expected = _Reference(gm).run(x, weight)
                    self.assertEqual(self._apply(gm).matched_count, 1)
                    torch.testing.assert_close(gm(x, weight), expected)

    def test_unsupported_graphs_remain_unchanged(self):
        for kwargs in (
            {"bias": True},
            {"extra_gather_user": True},
            {"tokens": 13, "extra_slice_user": True},
            {"dim": 1},
            {"tp": 4},
            {"group": "other_group"},
            {"k": 128},
            {"k": 65535},
            {"dtype": torch.float32},
            {"weight_dtype": torch.float16},
            {"noncontiguous": True},
            {"weight_noncontiguous": True},
            {"tokens": 13, "slice_dim": 1},
            {"tokens": 13, "start": 1},
            {"tokens": 13, "step": 2},
        ):
            with self.subTest(**kwargs):
                gm = _graph(**kwargs)
                before = str(gm.graph)
                self.assertEqual(self._apply(gm).matched_count, 0)
                self.assertEqual(str(gm.graph), before)

    def test_reduction_dimension_boundaries(self):
        for k in (256, 65534):
            with self.subTest(k=k):
                self.assertEqual(self._apply(_graph(k=k)).matched_count, 1)

    def test_unsupported_tp_skips_communicator_lookup(self):
        for tp in (1, 3, 16):
            with (
                self.subTest(tp=tp),
                patch(f"{MODULE}.get_tp_group") as group,
                patch(f"{MODULE}.get_tensor_model_parallel_world_size", return_value=tp),
            ):
                fusion = AllGatherMatmulFusionPass(VllmConfig())
                self.assertFalse(fusion.is_applicable_for_range(MagicMock()))
                group.assert_not_called()

    def test_multiple_matchers_and_backend_deduplication(self):
        first = self._make_pass()
        second = self._make_pass()
        self.assertEqual(base_pattern.nge.register_replacement.call_count, 2)
        for fusion in (first, second):
            gm = _graph()
            fusion(gm.graph)
            self.assertEqual(fusion.matched_count, 1)
        self._make_pass(group="tp:second")
        self.assertEqual(base_pattern.nge.register_replacement.call_count, 4)

    def test_nge_uses_scalar_and_safety_checks(self):
        backend = pm.PatternMatcherPass()

        def register_backend(**kwargs):
            pm.register_replacement(**kwargs, trace_fn=pm.fwd_only, pass_dicts=backend)

        with patch.object(base_pattern.nge, "register_replacement", side_effect=register_backend):
            self._make_pass()
        for tokens in (None, 3, 11, 15):
            with self.subTest(tokens=tokens):
                gm = _graph(tokens=tokens)
                self.assertEqual(backend.apply(gm.graph), 1)
                gm.graph.lint()
                self.assertNotIn(GATHER, _targets(gm))
        self.assertEqual(backend.apply(_graph(noncontiguous=True).graph), 0)

    def test_nge_respects_stream_scope(self):
        with patch.object(AllGatherMatmulPattern, "get_extra_stream_scope_check", return_value=lambda match: False):
            self._make_pass()
        kwargs = base_pattern.nge.register_replacement.call_args_list[0].kwargs
        self.assertFalse(kwargs["extra_check"](MagicMock()))


def test_base_adaptor_forwards_all_gather_contract():
    x, weight = torch.empty(8, K), torch.empty(K, N)
    with patch("vllm_ascend.device.device_op.torch_npu.npu_all_gather_base_mm", create=True) as op:
        result = BaseDeviceAdaptor.npu_all_gather_base_mm(x, weight, HCOM, TP_SIZE)
    op.assert_called_once_with(x, weight, HCOM, TP_SIZE, gather_index=0, gather_output=False, comm_mode="ai_cpu")
    assert result is op.return_value
