"""
DIVE-26: keep generation logits in model dtype for DIVE-25.

Principle:
  In greedy short-answer generation, logits are only ranked to choose the next
  token. If `lm_head` already produced fp16 logits, casting them to fp32 does
  not recover information; it only adds bandwidth and allocation work.

Modification:
  DIVE-25 -> DIVE-26. Keep selector math, merge math, pruning layer, refresh
  schedule, halo behavior, last-logit projection, no-padding mask fast path, and
  all DIVE parameters unchanged. The only runtime change is to keep logits in
  their native dtype during cached generation when no labels are requested.

Hypothesis:
  Avoiding the final fp16-to-fp32 logits cast reduces generation overhead without
  changing greedy decisions, because the values being compared are exactly the
  same fp16 outputs from the vocabulary head.

Free parameters:
  None new.

Limit cases:
  Training and scoring paths with `labels` still cast logits to fp32 for loss
  computation. Non-cached calls keep the original full-precision output behavior.
"""

from .dive25 import DIVE25


class DIVE26(DIVE25):
    """DIVE-26: DIVE-25 with native-dtype generation logits."""

    _DIVE_NAME = "dive26"
    _KEEP_GENERATION_LOGITS_DTYPE = True
