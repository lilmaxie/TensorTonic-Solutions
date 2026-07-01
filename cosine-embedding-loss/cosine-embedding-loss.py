import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    
    cos_sim = dot_product/(norm_x1 * norm_x2)

    if label == 1:
        L = 1.0-cos_sim
    elif label == -1:
        L = np.maximum(0.0, cos_sim - margin)
    else:
        raise ValueError("Error value")

    return float(L)