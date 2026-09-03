"""DIVE visual-token reduction model.

This module contains the consolidated final DIVE implementation. The former
DIVE-10 through DIVE-35 files represented incremental variants of the same
model; their behavior is kept here without the versioned inheritance chain.
"""

import math
import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from ..configuration_llama import LlamaConfig
from ..modeling_llama_self import LlamaModel
from .utils import compute_keep_count


_EPS_JIT = 1e-8


@torch.jit.script
def _normalize_unselected_jit(
    score: torch.Tensor, selected_mask: torch.Tensor, eps: float = _EPS_JIT
) -> torch.Tensor:
    valid_mask = ~selected_mask
    if not valid_mask.any():
        return torch.zeros_like(score)
    valid_scores = score[valid_mask]
    lo, hi = valid_scores.min(), valid_scores.max()
    if (hi - lo).abs() < eps:
        out = torch.zeros_like(score)
        out[valid_mask] = 1.0
        return out
    out = (score - lo) / (hi - lo + eps)
    out.masked_fill_(selected_mask, 0.0)
    return out.clamp_(0.0, 1.0)


@torch.jit.script
def _effective_support_jit(score: torch.Tensor, eps: float = _EPS_JIT) -> torch.Tensor:
    mass = score.clamp_min(0.0)
    n = mass.numel()
    if n <= 1:
        return torch.tensor(float(n), dtype=mass.dtype, device=mass.device)
    l1 = mass.sum()
    l2_sq = mass.square().sum().clamp_min(eps)
    return (l1 * l1 / l2_sq).clamp(1.0, float(n))


@torch.jit.script
def _dive15_select_no_text_jit(
    h_residual: torch.Tensor,
    source_energy_raw: torch.Tensor,
    keep_count: int,
    ortho_eta: float,
    eps: float,
) -> torch.Tensor:
    img_len = source_energy_raw.numel()
    selected_mask = torch.zeros(img_len, device=source_energy_raw.device, dtype=torch.bool)
    selected = torch.empty(keep_count, device=source_energy_raw.device, dtype=torch.long)
    source_decay = 2.0 * ortho_eta - ortho_eta * ortho_eta
    one_bool = torch.ones(1, device=source_energy_raw.device, dtype=torch.bool)

    for step in range(keep_count):
        score = source_energy_raw.clone()
        score.masked_fill_(selected_mask, float("-inf"))
        j_star = score.argmax()
        j_star1 = j_star.unsqueeze(0)
        selected_mask.scatter_(0, j_star1, one_bool)
        selected[step] = j_star

        v = h_residual.index_select(0, j_star1).squeeze(0)
        v_norm_sq = source_energy_raw.index_select(0, j_star1).squeeze(0)
        denom = v_norm_sq.clamp_min(eps)
        dot_h = h_residual.mv(v)
        a = (dot_h / denom).clamp_min_(0.0)

        h_residual.addcmul_(a.unsqueeze(1), v.unsqueeze(0), value=-ortho_eta)
        source_energy_raw.add_((a * a).mul_(-source_decay) * v_norm_sq).clamp_min_(0.0)

    return selected


@torch.jit.script
def _dive31_select_text_no_dump_jit(
    h_residual: torch.Tensor,
    source_energy_raw: torch.Tensor,
    C: torch.Tensor,
    keep_count: int,
    task_topm: int,
    task_lambda: float,
    ortho_eta: float,
    task_eta: float,
    refresh_stride: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    img_len = source_energy_raw.numel()
    selected_mask = torch.zeros(img_len, device=source_energy_raw.device, dtype=torch.bool)
    selected = torch.empty(keep_count, device=source_energy_raw.device, dtype=torch.long)
    task_weight = torch.ones_like(source_energy_raw)
    initial_task_align = torch.empty_like(source_energy_raw)
    initial_support = torch.zeros((), dtype=source_energy_raw.dtype, device=source_energy_raw.device)

    source_decay = 2.0 * ortho_eta - ortho_eta * ortho_eta
    cross_eta = ortho_eta * task_eta
    refresh_count = 0
    one_bool = torch.ones(1, device=source_energy_raw.device, dtype=torch.bool)

    for step in range(keep_count):
        if step % refresh_stride == 0:
            h_norm = source_energy_raw.sqrt().clamp_min(eps)
            task_token_score = C.clamp_min(0.0) / h_norm.unsqueeze(0)
            raw_task_align = task_token_score.topk(task_topm, dim=0, sorted=False).values.mean(dim=0)
            task_align = _normalize_unselected_jit(raw_task_align, selected_mask, eps)
            if refresh_count == 0:
                initial_task_align.copy_(task_align)
                initial_support.copy_(_effective_support_jit(raw_task_align, eps))
            task_weight = torch.exp(task_lambda * task_align)
            refresh_count += 1

        score = source_energy_raw * task_weight
        score.masked_fill_(selected_mask, float("-inf"))
        j_star = score.argmax()
        j_star1 = j_star.unsqueeze(0)
        selected_mask.scatter_(0, j_star1, one_bool)
        selected[step] = j_star

        v = h_residual.index_select(0, j_star1).squeeze(0)
        v_norm_sq = source_energy_raw.index_select(0, j_star1).squeeze(0)
        denom = v_norm_sq.clamp_min(eps)
        dot_h = h_residual.mv(v)
        a = (dot_h / denom).clamp_min_(0.0)
        dot_q = C.index_select(1, j_star1).squeeze(1)
        b = (dot_q / denom).clamp_min_(0.0)

        h_residual.addcmul_(a.unsqueeze(1), v.unsqueeze(0), value=-ortho_eta)
        source_energy_raw.add_((a * a).mul_(-source_decay) * v_norm_sq).clamp_min_(0.0)
        C.addcmul_(dot_q.unsqueeze(1), a.unsqueeze(0), value=-ortho_eta)
        C.addcmul_(b.unsqueeze(1), dot_h.unsqueeze(0), value=-task_eta)
        C.add_(b.unsqueeze(1) * a.unsqueeze(0) * (cross_eta * v_norm_sq))

    return selected, initial_task_align, initial_support, refresh_count


class DIVE(LlamaModel):
    """Consolidated final DIVE implementation."""

    _EPS = 1e-8
    _DIVE_NAME = "dive"
    _RETURN_LAYER_AUX = True
    _ASSUME_NO_PADDING = True
    _LAST_LOGIT_ONLY = True
    _KEEP_GENERATION_LOGITS_DTYPE = True

    def __init__(self, config: LlamaConfig):
        self.last_attention = None
        super().__init__(config)
        self.config = config

    def _profile_mark(self, enabled, device, marks, name):
        if not enabled:
            return
        if device.type == "cuda":
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            marks[name] = e
        else:
            marks[name] = time.perf_counter()

    def _profile_elapsed(self, enabled, device, marks, start, end):
        if not enabled or start not in marks or end not in marks:
            return 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            return float(marks[start].elapsed_time(marks[end]))
        return (marks[end] - marks[start]) * 1e3

    def _normalize_unselected(self, score, selected_mask):
        return _normalize_unselected_jit(score, selected_mask, self._EPS)

    def _effective_support(self, score):
        return _effective_support_jit(score, self._EPS)

    def _choose_text_source(self, dive_config, last_layer_state, img_start, img_end):
        suffix = last_layer_state[0][img_end:].float()
        prefix = last_layer_state[0][:img_start].float()
        mode = dive_config.get("text_mode", "suffix")
        if mode == "all":
            return torch.cat([prefix, suffix], dim=0)
        if mode == "prefix":
            return prefix if prefix.shape[0] > 0 else suffix
        return suffix if suffix.shape[0] > 0 else prefix

    def _get_refresh_stride(self, dive_config, text_len, img_len, keep_count):
        explicit = int(dive_config.get("task_refresh_stride", 0))
        if explicit > 0:
            return explicit, False
        side = int(math.sqrt(float(img_len)))
        if side * side == img_len and text_len > 3 * side:
            return 1, True
        return max(1, math.ceil(math.sqrt(float(keep_count)) / 2.0)), False

    def _index_mask(self, size, device, indices):
        mask = torch.zeros(size, device=device, dtype=torch.bool)
        mask[indices] = True
        return mask

    def _halo_neighbor_chunks(self, anchors, side):
        rows = torch.div(anchors, side, rounding_mode="floor")
        cols = anchors.remainder(side)

        neighbors = []
        if torch.any(rows > 0):
            neighbors.append(anchors[rows > 0] - side)
        if torch.any(rows < side - 1):
            neighbors.append(anchors[rows < side - 1] + side)
        if torch.any(cols > 0):
            neighbors.append(anchors[cols > 0] - 1)
        if torch.any(cols < side - 1):
            neighbors.append(anchors[cols < side - 1] + 1)
        return neighbors

    def _top_halo_candidates(self, initial_task_align, candidates, swap_n):
        n_cands = min(swap_n, candidates.numel())
        return candidates[
            initial_task_align[candidates].topk(n_cands, sorted=False).indices
        ]

    def _halo_droppable(self, selected_order, protected):
        reverse_order = selected_order.flip(0)
        return reverse_order[~protected[reverse_order]]

    def _build_keep_indices(self, img_start, img_len, seq_length, retained_idx, device):
        return torch.cat(
            (
                torch.arange(img_start, device=device),
                retained_idx,
                torch.arange(img_start + img_len, seq_length, device=device),
            )
        )

    def _semantic_halo_repair(
        self,
        retained_local: torch.Tensor,
        selected_order: torch.Tensor,
        initial_task_align: torch.Tensor,
        initial_support: torch.Tensor,
        img_len: int,
        keep_count: int,
    ) -> tuple[torch.Tensor, int, bool, float]:
        """Swap late tokens with semantically strong halo neighbors."""
        side = int(math.sqrt(float(img_len)))
        if side * side != img_len or keep_count <= 1:
            return retained_local, 0, False, 0.0

        support_value = float(initial_support.detach().cpu())
        gate_open = support_value > float(side) and support_value < float(keep_count)
        if not gate_open:
            return retained_local, 0, False, support_value

        swap_n = max(1, int(support_value / side))
        swap_n = min(swap_n, max(1, keep_count // 4))

        anchors = initial_task_align.topk(1, sorted=False).indices
        neighbors = self._halo_neighbor_chunks(anchors, side)
        if not neighbors:
            return retained_local, 0, True, support_value

        selected_mask = self._index_mask(img_len, retained_local.device, retained_local)
        candidates = torch.unique(torch.cat(neighbors))
        candidates = candidates[~selected_mask[candidates]]
        if candidates.numel() == 0:
            return retained_local, 0, True, support_value

        top_candidates = self._top_halo_candidates(
            initial_task_align, candidates, swap_n
        )
        protected = self._index_mask(img_len, retained_local.device, anchors)
        droppable = self._halo_droppable(selected_order, protected)
        if droppable.numel() == 0:
            return retained_local, 0, True, support_value

        n_swaps = min(top_candidates.numel(), droppable.numel())
        repaired_mask = selected_mask.clone()
        repaired_mask[droppable[:n_swaps]] = False
        repaired_mask[top_candidates[:n_swaps]] = True
        repaired = repaired_mask.nonzero(as_tuple=False).flatten()
        return repaired, n_swaps, True, support_value

    def _dive_dump_maps(
        self,
        dive_dump_dir,
        dump_index,
        img_start,
        img_len,
        keep_count,
        task_lambda,
        raw_image_energy,
        initial_raw_task_align,
        initial_raw_fused_score,
        initial_source_energy,
        initial_task_align,
        initial_fused_score,
        retained_local,
    ):
        os.makedirs(dive_dump_dir, exist_ok=True)
        side = int(math.sqrt(float(img_len)))
        shape = (side, side) if side * side == img_len else (int(img_len),)

        def to_numpy(x):
            return x.detach().float().cpu().numpy().reshape(shape).astype(np.float32)

        np.savez_compressed(
            os.path.join(dive_dump_dir, f"{self._DIVE_NAME}_maps_{dump_index:05d}.npz"),
            visual=to_numpy(raw_image_energy),
            text=to_numpy(initial_raw_task_align),
            fused=to_numpy(initial_raw_fused_score),
            normalized_visual=to_numpy(initial_source_energy),
            normalized_text=to_numpy(initial_task_align),
            normalized_fused=to_numpy(initial_fused_score),
            retained=retained_local.detach().cpu().numpy().astype(np.int64),
            img_start=np.asarray(int(img_start), dtype=np.int64),
            img_len=np.asarray(int(img_len), dtype=np.int64),
            keep_count=np.asarray(int(keep_count), dtype=np.int64),
            task_lambda=np.asarray(float(task_lambda), dtype=np.float32),
        )

    def _select_tokens_full(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states=None):
        dive_config = config.DIVE_config
        img_start = dive_config["image_token_start_index"]
        img_len = dive_config["image_token_length"]
        max_tok = dive_config["max_num_trunction"]
        ratio = dive_config["reduction_ratio"]

        if config.text_length is not None:
            img_len = last_layer_state.shape[1] - config.text_length
        img_end = img_start + img_len
        device = last_layer_state.device

        keep_count = compute_keep_count(max_tok, img_len, ratio)
        keep_count = min(keep_count, img_len)
        if keep_count <= 0:
            return torch.empty(0, device=device, dtype=torch.long)

        profile_enabled = bool(dive_config.get("profile_selector", False))
        marks = {}
        self._profile_mark(profile_enabled, device, marks, "start")

        raw_image_states = last_layer_state[0][img_start:img_end].float()
        raw_image_energy = (raw_image_states * raw_image_states).sum(dim=-1).clamp_min_(0.0)
        image_states = F.normalize(raw_image_states, dim=-1)
        text_source = self._choose_text_source(dive_config, last_layer_state, img_start, img_end)
        has_text = text_source.shape[0] > 0
        text_states = F.normalize(text_source, dim=-1) if has_text else None

        h_residual = image_states.clone()
        source_energy_raw = (h_residual * h_residual).sum(dim=-1).clamp_min_(0.0)
        C = text_states @ h_residual.T if has_text else None

        selected_mask = torch.zeros(img_len, device=device, dtype=torch.bool)
        selected = torch.empty(keep_count, device=device, dtype=torch.long)

        task_topm = int(dive_config.get("task_topm", 3))
        task_lambda = float(dive_config.get("task_lambda", 1.5))
        ortho_eta = float(dive_config.get("ortho_eta", 0.8))
        task_eta = float(dive_config.get("task_eta", 0.8))
        source_decay = 2.0 * ortho_eta - ortho_eta * ortho_eta
        cross_eta = ortho_eta * task_eta

        task_weight = None
        initial_task_align = None
        initial_raw_task_align = None
        initial_raw_fused_score = None
        initial_source_energy = None
        initial_fused_score = None
        initial_support = None
        refresh_count = 0
        refresh_stride, three_row_exact_refresh = self._get_refresh_stride(
            dive_config, int(text_source.shape[0]), int(img_len), int(keep_count)
        )
        if has_text and C is not None:
            task_topm = max(1, min(task_topm, C.shape[0]))

        self._profile_mark(profile_enabled, device, marks, "init")

        # Keep the OMP loop inline so DIVE-10 preserves DIVE-9's dispatch shape.
        for step in range(keep_count):
            if has_text and C is not None:
                if task_weight is None or step % refresh_stride == 0:
                    h_norm = source_energy_raw.sqrt().clamp_min(self._EPS)
                    task_token_score = C.clamp_min(0.0) / h_norm.unsqueeze(0)
                    raw_task_align = task_token_score.topk(
                        task_topm, dim=0, sorted=False
                    ).values.mean(dim=0)
                    task_align = self._normalize_unselected(raw_task_align, selected_mask)
                    if initial_task_align is None:
                        initial_task_align = task_align.clone()
                        initial_raw_task_align = raw_task_align.clone()
                        initial_raw_fused_score = raw_image_energy * torch.exp(task_lambda * raw_task_align)
                        initial_source_energy = source_energy_raw.clone()
                        initial_fused_score = source_energy_raw * torch.exp(task_lambda * task_align)
                        initial_support = self._effective_support(raw_task_align)
                    task_weight = torch.exp(task_lambda * task_align)
                    refresh_count += 1
                score = source_energy_raw * task_weight
            else:
                # Clone so masked_fill_ does not corrupt source_energy_raw.
                score = source_energy_raw.clone()

            score.masked_fill_(selected_mask, float("-inf"))
            j_star = score.argmax()
            j_star1 = j_star.unsqueeze(0)

            selected_mask.index_fill_(0, j_star1, True)
            selected[step] = j_star

            v = h_residual.index_select(0, j_star1).squeeze(0)
            v_norm_sq = source_energy_raw.index_select(0, j_star1).squeeze(0)

            denom = v_norm_sq.clamp_min(self._EPS)
            dot_h = h_residual.mv(v)
            a = (dot_h / denom).clamp_min_(0.0)

            if has_text and C is not None:
                dot_q = C.index_select(1, j_star1).squeeze(1)
                b = (dot_q / denom).clamp_min_(0.0)

            h_residual.addcmul_(a.unsqueeze(1), v.unsqueeze(0), value=-ortho_eta)
            source_energy_raw.add_((a * a).mul_(-source_decay) * v_norm_sq).clamp_min_(0.0)

            if has_text and C is not None:
                C.addcmul_(dot_q.unsqueeze(1), a.unsqueeze(0), value=-ortho_eta)
                C.addcmul_(b.unsqueeze(1), dot_h.unsqueeze(0), value=-task_eta)
                C.add_(b.unsqueeze(1) * a.unsqueeze(0) * (cross_eta * v_norm_sq))

        retained_local = selected
        halo_swaps = 0
        support_gate_open = False
        support_value = 0.0
        if initial_task_align is not None and initial_support is not None:
            (
                retained_local,
                halo_swaps,
                support_gate_open,
                support_value,
            ) = self._semantic_halo_repair(
                retained_local,
                retained_local,
                initial_task_align,
                initial_support,
                img_len,
                int(keep_count),
            )

        self._profile_mark(profile_enabled, device, marks, "done")
        if profile_enabled:
            self.last_selector_profile = {
                "method": self._DIVE_NAME,
                "keep_count": int(keep_count),
                "img_len": int(img_len),
                "text_len": int(text_source.shape[0]),
                "task_refresh_stride": int(refresh_stride),
                "task_refresh_count": int(refresh_count),
                "three_row_exact_refresh": bool(three_row_exact_refresh),
                "semantic_halo_swaps": int(halo_swaps),
                "semantic_support": support_value,
                "support_gate_open": support_gate_open,
                "support_gate_policy": "budget_upper_bound",
                "no_item_sync": True,
                "init_ms": self._profile_elapsed(profile_enabled, device, marks, "start", "init"),
                "loop_ms": self._profile_elapsed(profile_enabled, device, marks, "init", "done"),
                "total_ms": self._profile_elapsed(profile_enabled, device, marks, "start", "done"),
            }

        dive_dump_dir = dive_config.get("dive_dump_dir")
        if dive_dump_dir and initial_raw_task_align is not None:
            dump_index = int(getattr(self, "_dive_dump_index", 0))
            self._dive_dump_index = dump_index + 1
            self._dive_dump_maps(
                dive_dump_dir,
                dump_index,
                img_start,
                img_len,
                keep_count,
                task_lambda,
                raw_image_energy,
                initial_raw_task_align,
                initial_raw_fused_score,
                initial_source_energy,
                initial_task_align,
                initial_fused_score,
                retained_local,
            )

        retained_idx = retained_local.long() + img_start
        return retained_idx.sort().values

    @staticmethod
    def _effective_image_length(config: LlamaConfig, last_layer_state: torch.Tensor) -> int:
        img_len = int(config.DIVE_config["image_token_length"])
        if config.text_length is not None:
            img_len = int(last_layer_state.shape[1] - config.text_length)
        return img_len

    @staticmethod
    def _is_square_grid(img_len: int) -> bool:
        if img_len <= 0:
            return False
        side = math.isqrt(img_len)
        return side * side == img_len

    def _select_tokens(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states=None):
        """Select retained image tokens using the final geometry-gated path."""
        img_len = self._effective_image_length(config, last_layer_state)
        if self._is_square_grid(img_len) or config.DIVE_config.get("dive_dump_dir"):
            return self._select_tokens_full(config, last_layer_state, any_states)
        return self._select_tokens_no_dump(config, last_layer_state, any_states)

    def _select_tokens_no_dump(
        self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states=None
    ):
        """Select non-square layouts without allocating dump-only diagnostics."""
        dive_config = config.DIVE_config
        img_start = dive_config["image_token_start_index"]
        img_len = self._effective_image_length(config, last_layer_state)
        max_tok = dive_config["max_num_trunction"]
        ratio = dive_config["reduction_ratio"]
        img_end = img_start + img_len
        device = last_layer_state.device

        keep_count = min(compute_keep_count(max_tok, img_len, ratio), img_len)
        if keep_count <= 0:
            return torch.empty(0, device=device, dtype=torch.long)

        profile_enabled = bool(dive_config.get("profile_selector", False))
        marks = {}
        self._profile_mark(profile_enabled, device, marks, "start")

        raw_image_states = last_layer_state[0][img_start:img_end].float()
        image_states = F.normalize(raw_image_states, dim=-1)
        text_source = self._choose_text_source(dive_config, last_layer_state, img_start, img_end)
        has_text = text_source.shape[0] > 0
        text_states = F.normalize(text_source, dim=-1) if has_text else None
        h_residual = image_states.clone()
        source_energy_raw = (h_residual * h_residual).sum(dim=-1).clamp_min_(0.0)
        C = text_states @ h_residual.T if has_text else None

        task_topm = int(dive_config.get("task_topm", 3))
        task_lambda = float(dive_config.get("task_lambda", 1.5))
        ortho_eta = float(dive_config.get("ortho_eta", 0.8))
        task_eta = float(dive_config.get("task_eta", 0.8))
        refresh_count = 0
        refresh_stride, three_row_exact_refresh = self._get_refresh_stride(
            dive_config, int(text_source.shape[0]), img_len, keep_count
        )
        if has_text and C is not None:
            task_topm = max(1, min(task_topm, C.shape[0]))

        self._profile_mark(profile_enabled, device, marks, "init")
        initial_task_align = None
        initial_support = None
        if has_text and C is not None:
            (
                selected,
                initial_task_align,
                initial_support,
                refresh_count,
            ) = _dive31_select_text_no_dump_jit(
                h_residual,
                source_energy_raw,
                C,
                int(keep_count),
                int(task_topm),
                task_lambda,
                ortho_eta,
                task_eta,
                int(refresh_stride),
                self._EPS,
            )
        else:
            selected = _dive15_select_no_text_jit(
                h_residual, source_energy_raw, int(keep_count), ortho_eta, self._EPS
            )

        retained_local = selected
        halo_swaps = 0
        support_gate_open = False
        support_value = 0.0
        if initial_task_align is not None and initial_support is not None:
            (
                retained_local,
                halo_swaps,
                support_gate_open,
                support_value,
            ) = self._semantic_halo_repair(
                retained_local,
                retained_local,
                initial_task_align,
                initial_support,
                img_len,
                keep_count,
            )

        self._profile_mark(profile_enabled, device, marks, "done")
        if profile_enabled:
            self.last_selector_profile = {
                "method": self._DIVE_NAME,
                "keep_count": int(keep_count),
                "img_len": int(img_len),
                "text_len": int(text_source.shape[0]),
                "task_refresh_stride": int(refresh_stride),
                "task_refresh_count": int(refresh_count),
                "three_row_exact_refresh": bool(three_row_exact_refresh),
                "semantic_halo_swaps": int(halo_swaps),
                "semantic_support": support_value,
                "support_gate_open": support_gate_open,
                "support_gate_policy": "budget_upper_bound",
                "no_item_sync": True,
                "scripted_loop": True,
                "dump_diagnostics": False,
                "init_ms": self._profile_elapsed(profile_enabled, device, marks, "start", "init"),
                "loop_ms": self._profile_elapsed(profile_enabled, device, marks, "init", "done"),
                "total_ms": self._profile_elapsed(profile_enabled, device, marks, "start", "done"),
            }

        return (retained_local.long() + img_start).sort().values

    def _merge_dropped_tokens(
        self,
        hidden_states,
        keep_indices,
        retained_idx,
        img_start,
        img_len,
        last_layer_state,
    ):
        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_dim = hidden_states.shape[-1]
        gamma = self.config.DIVE_config.get("merge_gamma", 0.2)
        if gamma == 0.0:
            return hidden_states[:, keep_indices, :]
        img_end = img_start + img_len
        all_img_idx = torch.arange(img_start, img_end, device=device)
        keep_mask = torch.zeros(img_len, device=device, dtype=torch.bool)
        keep_mask[(retained_idx - img_start).long()] = True
        dropped_idx = all_img_idx[~keep_mask]
        if dropped_idx.numel() == 0:
            return hidden_states[:, keep_indices, :]
        dropped_vecs = F.normalize(last_layer_state[0][dropped_idx], dim=-1)
        kept_vecs = F.normalize(last_layer_state[0][retained_idx], dim=-1)
        assignments = torch.mm(dropped_vecs, kept_vecs.T).argmax(dim=1)
        dropped_hidden = hidden_states[0][dropped_idx]
        merge_sum = torch.zeros(len(retained_idx), hidden_dim, device=device, dtype=dtype)
        merge_cnt = torch.zeros(len(retained_idx), device=device, dtype=dtype)
        merge_sum.scatter_add_(
            0, assignments.unsqueeze(1).expand(-1, hidden_dim), dropped_hidden
        )
        merge_cnt.scatter_add_(
            0,
            assignments,
            torch.ones(dropped_idx.numel(), device=device, dtype=dtype),
        )
        merged = (
            hidden_states[0][retained_idx]
            + gamma * merge_sum / (merge_cnt.unsqueeze(1) + 1e-6)
        )
        hidden_states = hidden_states.clone()
        hidden_states[0][retained_idx] = merged
        return hidden_states[:, keep_indices, :]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            self._ASSUME_NO_PADDING
            and attention_mask is not None
            and attention_mask.dim() == 2
            and not output_attentions
        ):
            attention_mask = None

        if self._use_flash_attention_2:
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        prev_layer_outputs = None

        assert batch_size == 1, "batch_size > 1 not supported"

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                dive_config = self.config.DIVE_config
                if dive_config is not None:
                    prune_layer = dive_config["K"]
                    img_start = dive_config["image_token_start_index"]
                    img_len = dive_config["image_token_length"]
                    if (
                        prev_layer_outputs is not None
                        and decoder_layer.self_attn.layer_idx == prune_layer
                        and seq_length > 1
                    ):
                        device = hidden_states.device
                        last_layer_state = self.norm(prev_layer_outputs[0])
                        if self.config.text_length is not None:
                            img_len = last_layer_state.shape[1] - self.config.text_length
                        retained_idx = self._select_tokens(
                            self.config, last_layer_state, prev_layer_outputs
                        )
                        keep_indices = self._build_keep_indices(
                            img_start, img_len, seq_length, retained_idx, device
                        )
                        hidden_states = self._merge_dropped_tokens(
                            hidden_states,
                            keep_indices,
                            retained_idx,
                            img_start,
                            img_len,
                            last_layer_state,
                        )
                        if attention_mask is not None:
                            attention_mask = attention_mask[
                                :, :, : hidden_states.shape[1], : hidden_states.shape[1]
                            ]
                        position_ids = keep_indices.unsqueeze(0)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    return_layer_aux=self._RETURN_LAYER_AUX,
                )
            hidden_states = layer_outputs[0]
            prev_layer_outputs = layer_outputs
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache and not bool(getattr(self, "_RETURN_DYNAMIC_CACHE", False))
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
