# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse sequence-parallel all-gather with a bias-free column projection."""

from __future__ import annotations

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import Match, PatternMatcherPass
from vllm.compilation.passes.inductor_pass import enable_fake_mode
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import logger

from vllm_ascend.compilation.passes import base_pattern
from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.device.device_op import DeviceOperator

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_MIN_REDUCE_DIM = 256
_MAX_REDUCE_DIM_EXCLUSIVE = 65535
_GATHER_OP = torch.ops.vllm.all_gather.default
_GEMM_OP = torch.ops.vllm.unquantized_gemm.default
_SLICE_OP = torch.ops.aten.slice.Tensor


def _arg(node: fx.Node, index: int, name: str, default=None):
    return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)


class _AllGatherMatmulPattern(BasePattern):
    def __init__(self, config: VllmConfig, tp_size: int, group_name: str, hcom_name: str):
        super().__init__(config)
        self.tp_size = tp_size
        self.group_name = group_name
        self.hcom_name = hcom_name

    def get_inputs(self) -> list[torch.Tensor]:
        dtype = self.dtype if self.dtype in _SUPPORTED_DTYPES else torch.bfloat16
        device = self.vllm_config.device_config.device if self.vllm_config.device_config else "cpu"
        return [
            torch.empty(8, 512, dtype=dtype, device=device),
            torch.empty(256, 512, dtype=dtype, device=device),
        ]

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.dtype}_{self.tp_size}_{self.group_name}_{self.hcom_name}"

    def _get_scalar_workaround(self):
        return None

    def _get_extra_check(self):
        def supported(match: Match) -> bool:
            gather = next((n for n in match.nodes if n.target is _GATHER_OP), None)
            gemm = next((n for n in match.nodes if n.target is _GEMM_OP), None)
            if gather is None or gemm is None:
                return False
            if (
                _arg(gather, 1, "dim") != 0
                or _arg(gather, 2, "world_size") != self.tp_size
                or _arg(gather, 3, "group_name") != self.group_name
                or _arg(gemm, 2, "bias") is not None
                or len(gather.users) != 1
            ):
                return False
            x, weight = gather.args[0], gemm.args[1]
            if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
                return False
            x_val, weight_val = x.meta.get("val"), weight.meta.get("val")
            if x_val is None or weight_val is None or x_val.dim() != 2 or weight_val.dim() != 2:
                return False
            if (
                x_val.dtype not in _SUPPORTED_DTYPES
                or x_val.dtype != weight_val.dtype
                or x_val.device != weight_val.device
                or not x_val.is_contiguous()
                or not weight_val.is_contiguous()
            ):
                return False
            # Keep the reduction dimension static; token rows may be symbolic.
            k = x_val.shape[1]
            if not isinstance(k, int) or not _MIN_REDUCE_DIM <= k < _MAX_REDUCE_DIM_EXCLUSIVE:
                return False
            if not isinstance(weight_val.shape[1], int) or weight_val.shape[1] != k:
                return False
            sliced = next((n for n in match.nodes if n.target is _SLICE_OP), None)
            return sliced is None or (
                _arg(sliced, 1, "dim", 0) == 0
                and _arg(sliced, 2, "start", 0) == 0
                and _arg(sliced, 4, "step", 1) == 1
                and len(sliced.users) == 1
            )

        return supported

    @enable_fake_mode
    def register(self, pm_pass: PatternMatcherPass) -> None:
        key = self.pattern_key()
        registered = base_pattern._registered_by_matcher.setdefault(pm_pass, set())
        pattern, replacement = self.get_pattern(), self.get_replacement()
        inputs = self.get_inputs()
        check = self._get_extra_check()
        scalar_workaround = self._get_scalar_workaround()
        if key not in registered:
            pm.register_replacement(
                pattern,
                replacement,
                inputs,
                pm.fwd_only,
                pm_pass,
                extra_check=check,
                scalar_workaround=scalar_workaround,
            )
            registered.add(key)
        if key not in base_pattern._registered_patterns:
            stream_check = self.get_extra_stream_scope_check()
            base_pattern.nge.register_replacement(
                search_fn=pattern,
                replace_fn=replacement,
                example_inputs=inputs,
                extra_check=lambda match: stream_check(match) and check(match),
                scalar_workaround=scalar_workaround,
            )
            base_pattern._registered_patterns.add(key)


class AllGatherMatmulPattern(_AllGatherMatmulPattern):
    def get_pattern(self):
        def pattern(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            gathered = torch.ops.vllm.all_gather.default(x, dim=0, world_size=self.tp_size, group_name=self.group_name)
            return torch.ops.vllm.unquantized_gemm.default(gathered, weight, None)

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return DeviceOperator.npu_all_gather_base_mm(
                x, weight.t(), self.hcom_name, self.tp_size, gather_output=False
            )[0]

        return replacement


class SlicedAllGatherMatmulPattern(_AllGatherMatmulPattern):
    """Move token-prefix unpadding after the fused projection."""

    def _get_scalar_workaround(self):
        return {"num_tokens": 13}

    def get_pattern(self):
        def pattern(x: torch.Tensor, weight: torch.Tensor, num_tokens: int) -> torch.Tensor:
            gathered = torch.ops.vllm.all_gather.default(x, dim=0, world_size=self.tp_size, group_name=self.group_name)
            return torch.ops.vllm.unquantized_gemm.default(gathered[:num_tokens], weight, None)

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, weight: torch.Tensor, num_tokens: int) -> torch.Tensor:
            output = DeviceOperator.npu_all_gather_base_mm(
                x, weight.t(), self.hcom_name, self.tp_size, gather_output=False
            )[0]
            return output[:num_tokens]

        return replacement


class AllGatherMatmulFusionPass(VllmInductorPass):
    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pattern_match_passes = PatternMatcherPass(pass_name="all_gather_matmul_fusion_pass")
        if self.tp_size not in _SUPPORTED_WORLD_SIZES:
            return
        tp_group = get_tp_group()
        backend = tp_group.device_group._get_backend(torch.device("npu"))
        hcom_name = backend.get_hccl_comm_name(tp_group.rank_in_group)
        for pattern_cls in (AllGatherMatmulPattern, SlicedAllGatherMatmulPattern):
            pattern_cls(config, self.tp_size, tp_group.unique_name, hcom_name).register(self.pattern_match_passes)

    def __call__(self, graph: fx.Graph) -> None:
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s all-gather-matmul patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.tp_size in _SUPPORTED_WORLD_SIZES
