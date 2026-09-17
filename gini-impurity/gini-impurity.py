import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    def _node_gini(labels: list) -> float:
        """Helper function to calculate Gini impurity for a single node"""
        n = len(labels)
        if n == 0:
            return 0.0
        
        _, counts = np.unique(labels, return_counts=True)
        
        probs = counts / n
        
        return float(1.0 - np.sum(probs ** 2))

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    gini_left = _node_gini(y_left)
    gini_right = _node_gini(y_right)

    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

    return float(weighted_gini)