import math


def compute_token_topk(max_num_trunction, img_len, pivot_image, pivot_text, reduction_ratio,
                       retain_token_num=None):
    """Per-pivot token budget for greedy DIVE-style selection.

    Used by DIVE variants that iterate over pivot tokens.
    `retain_token_num` is only supplied for LLaVA-Next, where the budget is
    derived from a pre-computed per-image token count rather than a fixed ratio.
    """
    if retain_token_num is not None:
        n_budget = retain_token_num
    elif max_num_trunction:
        n_budget = max_num_trunction
    else:
        n_budget = img_len * (1 - reduction_ratio)
    return math.ceil(n_budget // (pivot_image + pivot_text))


def compute_keep_count(max_tok, img_len, ratio):
    """Total number of image tokens to retain.

    Used by DIVE selectors that select a flat token budget rather than
    iterating over pivots.
    """
    k = int(max_tok) if max_tok else int(img_len * (1.0 - ratio))
    return max(1, min(k, img_len))
