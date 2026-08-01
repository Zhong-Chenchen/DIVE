"""
DIVE-22: geometry-gated halo bookkeeping for DIVE-15.

Principle:
  Semantic halo repair only applies when visual tokens form a square grid, so
  non-square LLaVA-Next layouts should not pay a GPU-to-CPU synchronization for
  halo diagnostics that cannot affect token selection.

Modification:
  DIVE-15 -> DIVE-22. Keep the selector loop, refresh schedule, merge path,
  forward path, and all DIVE parameters unchanged. The only change is in halo
  repair bookkeeping: check the square-grid and keep-count gates before reading
  `initial_support` on the CPU.

Hypothesis:
  LLaVA-1.6 uses 2880 visual tokens, which is not a square grid. DIVE-15
  therefore never performs halo swaps there, but still synchronizes to materialize
  the support value. Removing that dead synchronization should improve prefill
  latency without changing retained indices.

Free parameters:
  None new. DIVE-22 uses every inherited DIVE-15 `DIVE_config` key with the
  same defaults.

Limit cases:
  For square-grid inputs such as 576 tokens, DIVE-22 falls back to DIVE-15's
  original halo logic and support diagnostics. For non-square inputs, halo repair
  is impossible in both methods, and DIVE-22 returns the unchanged retained set
  without computing the unused support scalar.
"""

import math

import torch

from .dive15 import DIVE15


class DIVE22(DIVE15):
    """DIVE-22: DIVE-15 without dead halo CPU sync on non-square layouts."""

    _DIVE_NAME = "dive22"

    def _semantic_halo_repair(
        self,
        retained_local: torch.Tensor,
        selected_order: torch.Tensor,
        initial_task_align: torch.Tensor,
        initial_support: torch.Tensor,
        img_len: int,
        keep_count: int,
    ) -> tuple[torch.Tensor, int, bool, float]:
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
