# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel AllGather + MatMul for SP column projections."""

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather
from vllm_ascend.ops.linear_op import CustomColumnParallelOp
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


def _sequence_parallel_agmm(x: torch.Tensor, layer_name: str, output_size: int, tp_size: int) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    return layer.custom_op.run(x)


def _sequence_parallel_agmm_fake(x: torch.Tensor, layer_name: str, output_size: int, tp_size: int) -> torch.Tensor:
    return x.new_empty((x.shape[0] * tp_size, output_size))


# Keep the weight-format inspection and the ND shadow copy out of Dynamo; a
# dummy trace must not bake either decision into the graph.
direct_register_custom_op(
    op_name="ascend_sequence_parallel_agmm",
    op_func=_sequence_parallel_agmm,
    fake_impl=_sequence_parallel_agmm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


class SequenceParallelAllGatherMMOp(CustomColumnParallelOp):
    """INT8 W8A8_DYNAMIC column projection with the SP AllGather folded in.

    The reference chain all-gathers BF16 activations and per-token quantizes
    the full sequence afterwards. The fused chain quantizes the rank-local
    shard first and lets the kernel gather the INT8 rows together with their
    per-token scales. Per-token quantization is row-local, so both chains
    agree bit-exactly while the collective moves half the bytes.

    Installation transfers the caller's AllGather to this op: fused when the
    runtime gates pass, plain AllGather + quant MM otherwise. Both paths see
    the rank-local SP shard and return the full-sequence projection, so the
    caller never branches.

    Weight (and scale) references are read at execution time, after loading
    transformations (including the INT8 weight transpose and optional NZ
    cast). The fused kernel only accepts ND x2, so an NZ weight is cast back
    to a bit-exact ND shadow copy once and cached.
    """

    SUPPORTED_TP_SIZES = (2, 4, 8)
    # A3 TP8 paired measurements cross over between m=32 and m=64. Smaller
    # batches keep the original AG + MM path; the local shape is TP-consistent.
    MIN_M = 64
    MIN_K = 256
    MAX_K = 65535  # kernel contract: k in [256, 65535)

    def __init__(self, layer, *, min_tokens: int, max_tokens: int):
        super().__init__(layer)
        self.update_attrs()
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.output_size_per_partition = layer.output_size // layer.tp_size
        backend = self.comm_group.device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(self.tp_rank)
        self._nd_weight: torch.Tensor | None = None
        self._x2_scale: torch.Tensor | None = None
        self._operand_weight: torch.Tensor | None = None
        self._operand_scale: torch.Tensor | None = None
        self._operand_key: tuple | None = None

    @staticmethod
    def unsupported_reason(layer) -> str | None:
        """Collect the layer-level facts fixed after loading."""
        if layer.custom_op is not None:
            return "an existing custom linear op owns the communication"
        if layer.gather_output:
            return "requires a sharded (non-gathered) column projection output"
        if layer.bias is not None:
            return "only bias-free projections are supported"
        if layer.tp_size not in SequenceParallelAllGatherMMOp.SUPPORTED_TP_SIZES:
            return "requires TP size 2, 4 or 8"
        if not callable(getattr(torch_npu, "npu_all_gather_quant_mm", None)):
            return "requires the torch_npu AllGather quant MM interface"
        if not isinstance(layer.quant_method, AscendLinearMethod):
            return "requires the Ascend W8A8_DYNAMIC adapter"
        # Do not accept subclasses such as FP8/PDMix with different scale ABIs.
        if type(layer.quant_method.quant_method) is not AscendW8A8DynamicLinearMethod:
            return "requires the INT8 W8A8_DYNAMIC scheme"
        return None

    def apply_impl(self, input_: torch.Tensor) -> tuple[torch.Tensor, None]:
        output = torch.ops.vllm.ascend_sequence_parallel_agmm(
            input_, self.prefix, self.output_size_per_partition, self.tp_size
        )
        return output, None

    def run(self, input_: torch.Tensor) -> torch.Tensor:
        # The only fused/fallback branch; every gate lives in _can_fuse and
        # is evaluated per forward, before any collective.
        if not self._can_fuse(input_):
            input_parallel = sp_all_gather(input_)
            return self.quant_method.apply(self.layer, input_parallel, None)
        return self.run_fused(input_)

    def _can_fuse(self, x: torch.Tensor) -> bool:
        if not is_forward_context_available():
            return False
        # The MLP has no attention metadata of its own; every entry describes
        # the same batch, so any of them carries the prefill/decode split.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values()), None)
        # Missing metadata includes dummy/profile runs. Decode and mixed
        # batches stay on the existing path, including FULL_DECODE_ONLY capture.
        if metadata is None or getattr(metadata, "num_prefills", 0) <= 0 or getattr(metadata, "num_decodes", None) != 0:
            return False
        if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[0] < self.MIN_M:
            return False
        if not self.MIN_K <= x.shape[1] < self.MAX_K:
            return False
        # The model gates the full-sequence token count to the INT8 benefit
        # range; the padded shard is TP-consistent on every rank.
        if not self.min_tokens <= x.shape[0] * self.tp_size < self.max_tokens:
            return False
        return self._static_weight_ok(x.shape[1])

    def _static_weight_ok(self, k: int) -> bool:
        """Validate weight/scale and cache the fused x2 operands.

        Loading finishes before the first forward, so only a replaced or
        in-place-mutated weight/scale is revalidated. The ND shadow copy of an
        NZ weight is built at most once per weight identity.
        """
        weight = self.layer.weight
        scale = getattr(self.layer, "weight_scale_fp32", None)
        if scale is None:
            return False
        key = (weight._version, scale._version, k)
        if self._operand_key == key and self._operand_weight is weight and self._operand_scale is scale:
            return True
        basic = (
            weight.ndim == 2
            and weight.dtype == torch.int8
            and weight.shape == (k, self.output_size_per_partition)
            and weight.is_contiguous()
            and scale.dtype == torch.float32
            and scale.shape == (self.output_size_per_partition,)
        )
        if not basic:
            return False
        fmt = torch_npu.get_npu_format(weight)
        if fmt == ACL_FORMAT_FRACTAL_ND:
            nd_weight = weight
        elif fmt == ACL_FORMAT_FRACTAL_NZ:
            nd_weight = torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_ND)
        else:
            return False
        self._nd_weight = nd_weight
        self._x2_scale = scale.reshape(1, -1).contiguous()
        self._operand_weight = weight
        self._operand_scale = scale
        self._operand_key = key
        return True

    def run_fused(self, input_: torch.Tensor) -> torch.Tensor:
        """The fused branch without the runtime gate.

        Split from run() so startup warmup can invoke it explicitly; callers
        must guarantee TP consistency themselves (run() does via the gate).
        """
        logger.info_once("Layer AGMM fused path active")
        if self._operand_key is None and not self._static_weight_ok(input_.shape[1]):
            raise RuntimeError("Layer AGMM fused path requires supported INT8 weights")
        scheme = self.quant_method.quant_method
        prepared = scheme.prepare_input_for_overlap(self.layer, input_.contiguous())
        assert prepared is not None and prepared.scale is not None
        output, _, _ = torch_npu.npu_all_gather_quant_mm(
            prepared.quantized,
            self._nd_weight,
            self.hcom,
            self.tp_size,
            x1_scale=prepared.scale.reshape(-1, 1).contiguous(),
            x2_scale=self._x2_scale,
            # gather_output=False crashes the A3 CANN 9.2.1 kernel (the gather
            # result is written through a null pointer); the gathered INT8
            # copy is discarded instead.
            gather_output=True,
            y_dtype=prepared.output_dtype,
            comm_mode="aiv",
        )
        return output


def install_sequence_parallel_agmm(layer, *, sequence_parallel: bool, min_tokens: int, max_tokens: int) -> bool:
    """Transfer the caller's AllGather to a supported linear; False changes nothing.

    The enable conditions mirror the layer MMRS form: sequence parallelism on
    a node that serves prefill, no LoRA, no context parallelism, TP 2/4/8 and
    an INT8 W8A8_DYNAMIC scheme. Unsupported layers keep the plain
    AllGather + MM path with no ownership change.
    """
    if not sequence_parallel:
        return False
    config = get_current_vllm_config()
    if config.lora_config is not None:
        return False
    # Context-parallel weight/group adaptations have a separate ownership
    # contract. Only the original TP group is supported in this first backend.
    if (
        config.parallel_config.prefill_context_parallel_size != 1
        or config.parallel_config.decode_context_parallel_size != 1
    ):
        return False
    # Pure decode nodes (PD-disaggregated kv_consumer) never see a prefill
    # batch, so the fused path could never fire.
    transfer = config.kv_transfer_config
    if transfer is not None and transfer.is_kv_consumer and not transfer.is_kv_producer:
        return False
    if reason := SequenceParallelAllGatherMMOp.unsupported_reason(layer):
        logger.info_once("Layer AGMM disabled for %s: %s", layer.prefix, reason)
        return False
    context = config.compilation_config.static_forward_context
    existing = context.get(layer.prefix)
    if existing is not None and existing is not layer:
        raise ValueError(f"Duplicate layer name: {layer.prefix}")
    context[layer.prefix] = layer
    layer.custom_op = SequenceParallelAllGatherMMOp(layer, min_tokens=min_tokens, max_tokens=max_tokens)
    logger.info_once("Layer AGMM enabled for %s", layer.prefix)
    return True
