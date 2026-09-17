import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    def _entropy(labels: np.ndarray) -> float:
        """Helper to calculate Shannon Entropy for an array of labels."""
        n = len(labels)
        if n == 0:
            return 0.0

        _, counts = np.unique(labels, return_counts=True)
        probs = counts/n

        return float(-np.sum(probs * np.log2(probs)))

    y = np.asarray(y)
    mask = np.asarray(split_mask, dtype=bool)

    n_total = len(y)
    n_left = int(np.sum(mask))
    n_right = n_total - n_left

    if n_left == 0 or n_right == 0 or n_total == 0:
        return 0.0

    h_parent = _entropy(y)

    y_left = y[mask]
    y_right = y[~mask] # bitwise NOT operator

    h_left = _entropy(y_left)
    h_right = _entropy(y_right)

    ig = h_parent - (n_left/n_total)*h_left - (n_right/n_total)*h_right

    return float(ig)