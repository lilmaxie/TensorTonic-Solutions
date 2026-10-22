import numpy as np

def add_position_embedding(patches: np.ndarray,
                           pos_embed: np.ndarray) -> np.ndarray:
    """
    Returns the float64 tokens after adding position embeddings.
    """
    x = np.asarray(patches, dtype=np.float64)
    e = np.asarray(pos_embed, dtype=np.float64)

    # element-wise
    out = x + e

    return out