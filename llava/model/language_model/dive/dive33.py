"""
DIVE-33: geometry-gated no-dump selector fast path.

Principle:
  The DIVE-31 selector fast path is most useful on non-square visual layouts,
  where diagnostic maps are not consumed by the square-grid halo path. Regular
  square grids can keep DIVE-26's fully materialized selector path to avoid
  carrying a micro-optimization that did not help the v1.5 smoke runs.

Modification:
  DIVE-31 -> DIVE-33. Keep all DIVE parameters, selector math, refresh
  schedule, halo repair, merge math, pruning layer, last-logit projection,
  no-padding mask fast path, and native generation logits unchanged. The only
  runtime change is a geometry gate: non-square image-token lengths use
  DIVE-31's no-dump selector, while square image-token lengths use DIVE-26.

Hypothesis:
  The gate should preserve DIVE-31's observed v1.6 benefit on 2880-token
  stitched visual layouts while avoiding the v1.5 576-token square-grid
  slowdown, without changing retained indices inside either branch.

Free parameters:
  None new.

Limit cases:
  When the image-token length is a perfect square, DIVE-33 degenerates to
  DIVE-26. When it is not a perfect square and `dive_dump_dir` is absent,
  DIVE-33 degenerates to DIVE-31. If `dive_dump_dir` is set, DIVE-31's own
  dump-preserving fallback still applies.
"""

import math

import torch

from ..configuration_llama import LlamaConfig
from .dive26 import DIVE26
from .dive31 import DIVE31


class DIVE33(DIVE31):
    """DIVE-33: DIVE-31 gated by visual-token grid geometry."""

    _DIVE_NAME = "dive33"

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
        img_len = self._effective_image_length(config, last_layer_state)
        if self._is_square_grid(img_len):
            return DIVE26._select_tokens(self, config, last_layer_state, any_states)
        return super()._select_tokens(config, last_layer_state, any_states)
