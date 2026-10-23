import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    enc_out = np.asarray(encoder_output, dtype=np.float64)
    w_h = np.asarray(W_head, dtype=np.float64)

    # extract the [CLS] token specifically from sequence index 0: shape (B, D)
    h = enc_out[:, 0, :]

    # compute LayerNorm along the final hidden dimension axis (axis=-1)
    # use population variance (ddof=0) and epsilon = 1e-6
    mean = np.mean(h, axis=-1, keepdims=True)
    var = np.var(h, axis=-1, keepdims=True)
    h_norm = (h-mean) / np.sqrt(var + 1e-6)

    # linear projection to a C-class classifier: (B, D) @ (D, C) -> (B, C)
    logits = h_norm@w_h

    return logits