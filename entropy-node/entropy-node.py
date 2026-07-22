import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)

    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)

    probs = counts/len(y)

    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))

    return max(0.0, float(entropy))