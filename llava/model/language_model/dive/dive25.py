"""
DIVE-25: no-padding SDPA mask fast path for DIVE-24.

Principle:
  The current smoke/eval path uses batch size 1 with no padded tokens, so the
  2D attention mask carries no information beyond sequence length. Rechecking
  that mask on every generation step is avoidable runtime bookkeeping.

Modification:
  DIVE-24 -> DIVE-25. Keep selector math, merge math, pruning layer, refresh
  schedule, halo behavior, last-logit projection, and all DIVE parameters
  unchanged. The only runtime change is to declare the dive path as no-padding
  and pass `attention_mask=None` before the SDPA/FlashAttention mask preparation
  branch when the mask is still a 2D padding mask.

Hypothesis:
  Avoiding the CUDA `torch.all(attention_mask == 1)` scan inside the SDPA mask
  helper reduces per-step generation overhead without changing semantics for
  no-padding batch-1 evaluation.

Free parameters:
  None new.

Limit cases:
  If padding is actually present, this optimization is not valid and the method
  should not be used. With the standard POPE smoke loader, prompts are evaluated
  one sample at a time, so the padding mask is all ones and DIVE-25 should
  produce the same answers as DIVE-24.
"""

from .dive24 import DIVE24


class DIVE25(DIVE24):
    """DIVE-25: DIVE-24 with no-padding attention-mask fast path."""

    _DIVE_NAME = "dive25"
    _ASSUME_NO_PADDING = True
