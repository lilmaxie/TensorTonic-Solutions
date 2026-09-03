import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    out = np.array(X, dtype=float)

    # 1D array case
    if out.ndim == 1:
        valid_mask = ~np.isnan(out)

        if not np.any(valid_mask):
            fill_val = 0.0
        else:
            observed_vals = out[valid_mask]
            if strategy == "median":
                fill_val = float(np.median(observed_vals))
            else:
                fill_val = float(np.mean(observed_vals))

        out[~valid_mask] = fill_val

    # 2D array case - independent across columns
    elif out.ndim == 2:
        num_cols = out.shape[1]
        for col_idx in range(num_cols):
            col = out[:, col_idx]
            valid_mask = ~np.isnan(col)

            if not np.any(valid_mask):
                fill_val = 0.0
            else:
                observed_vals = col[valid_mask]
                if strategy == "median":
                    fill_val = float(np.median(observed_vals))
                else:
                    fill_val = float(np.mean(observed_vals))

            out[~valid_mask, col_idx] = fill_val

    return out