import numpy as np

def unet_output(features: np.ndarray, W_out: np.ndarray,
                b_out: np.ndarray) -> np.ndarray:
    """
    Returns float64 per-pixel class logits in NHWC layout.
    """
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(W_out, dtype=np.float64)
    b = np.asarray(b_out, dtype=np.float64)
    
    # (B, H, W, C) @ (C, K) -> (B, H, W, K)
    # (B, H, W, K) + (K,)   -> (B, H, W, K)
    return x@w + b