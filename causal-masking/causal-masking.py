import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores_arr = np.asarray(scores, dtype=np.float64)

    T = scores_arr.shape[-1]

    # create an upper triangular mask with k=1 (positions where j > i).
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    masked_scores = np.where(mask, mask_value, scores_arr)

    return masked_scores