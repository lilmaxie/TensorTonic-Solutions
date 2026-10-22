import numpy as np

def prepend_class_token(patches: np.ndarray,
                        cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    p = np.asarray(patches, dtype=np.float64)
    cls = np.asarray(cls_token, dtype=np.float64)

    # batch size B from tensor patches
    B = p.shape[0]

    # broadcast the cls_token from (1, 1, D) to (B, 1, D) along the batch axis.
    cls_broadcast = np.broadcast_to(cls, (B, 1, cls.shape[2]))

    # concat the cls_token to the sequence along the sequence axis (axis=1)
    out = np.concatenate([cls_broadcast, p], axis=1)

    return out