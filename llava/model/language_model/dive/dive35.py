"""
DIVE-35: monotonic keep-index construction on the DIVE-33 path.

Principle:
  DIVE selectors return retained visual token indices in ascending absolute
  order. Since prefix tokens are before the visual span and suffix tokens are
  after it, concatenating prefix + retained visual tokens + suffix already
  produces the causal sequence order required by the later decoder layers.

Modification:
  DIVE-33 -> DIVE-35. Keep the geometry-gated selector, refresh schedule,
  halo repair, merge math, pruning layer, last-logit projection, no-padding
  mask fast path, and native generation logits unchanged. The only runtime
  change is to skip the redundant final GPU `.sort()` in `_build_keep_indices`.

Hypothesis:
  The sort is pure bookkeeping after a selector that already returns sorted
  retained indices. Removing it should shave fixed prefill-pruning overhead
  without changing token order or generated answers.

Free parameters:
  None new.

Limit cases:
  DIVE-35 is equivalent to DIVE-33 whenever the inherited selector returns
  sorted retained indices, which DIVE-26/31/33 do before returning. Selectors
  that may return unsorted retained indices should keep the base implementation.
"""

import torch

from .dive33 import DIVE33


class DIVE35(DIVE33):
    """DIVE-35: DIVE-33 with redundant keep-index sort removed."""

    _DIVE_NAME = "dive35"

    def _build_keep_indices(self, img_start, img_len, seq_length, retained_idx, device):
        return torch.cat(
            (
                torch.arange(img_start, device=device),
                retained_idx,
                torch.arange(img_start + img_len, seq_length, device=device),
            )
        )
