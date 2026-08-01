"""
DIVE-24: generation last-logit projection for DIVE-22.

Principle:
  Autoregressive generation only consumes the final position's logits from the
  prefill pass. Projecting every retained prompt token through the vocabulary
  head is avoidable work when no labels are requested and cache is enabled.

Modification:
  DIVE-22 -> DIVE-24. Keep selector math, merge math, pruning layer, refresh
  schedule, halo behavior, decoder-layer outputs, and all DIVE parameters
  unchanged. The only runtime change is to request the causal LM wrapper to
  slice hidden states to the last position before `lm_head` during cached
  generation.

Hypothesis:
  On short-answer VQA generation, the full-sequence vocabulary projection is a
  fixed prefill-side cost that does not affect decoding decisions. Computing
  only the last logits should reduce latency without changing generated tokens.

Free parameters:
  None new.

Limit cases:
  Training or scoring paths with `labels` keep full logits. Calls without cache
  also keep full logits. The optimization is active only for DIVE-24 through
  `_LAST_LOGIT_ONLY=True`.
"""

from .dive22 import DIVE22


class DIVE24(DIVE22):
    """DIVE-24: DIVE-22 with generation-only last-position lm_head."""

    _DIVE_NAME = "dive24"
    _LAST_LOGIT_ONLY = True
