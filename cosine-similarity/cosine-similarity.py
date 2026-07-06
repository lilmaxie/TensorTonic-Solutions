import numpy as np

def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two vectors a and b.
    Returns: float
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    dot_product = np.dot(a, b)
    similarity = dot_product / (norm_a * norm_b)
    
    return float(similarity)