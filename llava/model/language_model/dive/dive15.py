"""
DIVE-15: TorchScript-dispatched DIVE-10 greedy loop.

Builds on DIVE-10 and keeps the per-token greedy OMP order, task-refresh
schedule, residual equations, semantic halo repair, dive-map dumping, merge
path, and LLaMA forward path unchanged. The single engineering change is to
move the token-by-token greedy loop into top-level TorchScript helpers.

Motivation:
  DIVE-10 is arithmetic-stable, but its inner loop still returns to Python for
  every retained token and launches many small tensor operations. Scripting the
  loop removes Python interpreter overhead while preserving the sequential
  greedy dependency that block dispatch variants changed.

Expected behavior:
  DIVE-15 should return the same retained indices as DIVE-10 for ordinary
  inputs. It does not batch token selection, does not change refresh cadence,
  and does not introduce new hyperparameters. Any differences should be limited
  to TorchScript/PyTorch execution details for the same tensor operations.
"""

import torch
import torch.nn.functional as F

from ..configuration_llama import LlamaConfig
from .dive10 import DIVE10, _effective_support_jit, _normalize_unselected_jit
from .utils import compute_keep_count


@torch.jit.script
def _dive15_select_text_jit(
    h_residual: torch.Tensor,
    source_energy_raw: torch.Tensor,
    raw_image_energy: torch.Tensor,
    C: torch.Tensor,
    keep_count: int,
    task_topm: int,
    task_lambda: float,
    ortho_eta: float,
    task_eta: float,
    refresh_stride: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    img_len = source_energy_raw.numel()
    selected_mask = torch.zeros(img_len, device=source_energy_raw.device, dtype=torch.bool)
    selected = torch.empty(keep_count, device=source_energy_raw.device, dtype=torch.long)
    task_weight = torch.ones_like(source_energy_raw)

    initial_task_align = torch.empty_like(source_energy_raw)
    initial_raw_task_align = torch.empty_like(source_energy_raw)
    initial_raw_fused_score = torch.empty_like(source_energy_raw)
    initial_source_energy = torch.empty_like(source_energy_raw)
    initial_fused_score = torch.empty_like(source_energy_raw)
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
                initial_raw_task_align.copy_(raw_task_align)
                initial_raw_fused_score.copy_(raw_image_energy * torch.exp(task_lambda * raw_task_align))
                initial_source_energy.copy_(source_energy_raw)
                initial_fused_score.copy_(source_energy_raw * torch.exp(task_lambda * task_align))
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

    return (
        selected,
        initial_task_align,
        initial_raw_task_align,
        initial_raw_fused_score,
        initial_source_energy,
        initial_fused_score,
        initial_support,
        refresh_count,
    )


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


class DIVE15(DIVE10):
    """DIVE-15: DIVE-10 with a scripted sequential greedy selector loop."""

    _DIVE_NAME = "dive15"

    def _select_tokens(self, config: LlamaConfig, last_layer_state: torch.Tensor, any_states=None):
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

        task_topm = int(dive_config.get("task_topm", 3))
        task_lambda = float(dive_config.get("task_lambda", 1.5))
        ortho_eta = float(dive_config.get("ortho_eta", 0.8))
        task_eta = float(dive_config.get("task_eta", 0.8))

        refresh_count = 0
        refresh_stride, three_row_exact_refresh = self._get_refresh_stride(
            dive_config, int(text_source.shape[0]), int(img_len), int(keep_count)
        )
        if has_text and C is not None:
            task_topm = max(1, min(task_topm, C.shape[0]))

        self._profile_mark(profile_enabled, device, marks, "init")

        initial_task_align = None
        initial_raw_task_align = None
        initial_raw_fused_score = None
        initial_source_energy = None
        initial_fused_score = None
        initial_support = None

        if has_text and C is not None:
            (
                selected,
                initial_task_align,
                initial_raw_task_align,
                initial_raw_fused_score,
                initial_source_energy,
                initial_fused_score,
                initial_support,
                refresh_count,
            ) = _dive15_select_text_jit(
                h_residual,
                source_energy_raw,
                raw_image_energy,
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
                h_residual,
                source_energy_raw,
                int(keep_count),
                ortho_eta,
                self._EPS,
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
                "scripted_loop": True,
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
