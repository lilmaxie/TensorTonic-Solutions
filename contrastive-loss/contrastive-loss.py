import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y)

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must only contain values in [0, 1]")

    dist = np.linalg.norm(a-b, axis=-1)

    pos_loss = y * (dist ** 2)
    neg_loss = (1-y) * (np.maximum(0.0, margin - dist) ** 2)
    loss_per_sample = pos_loss + neg_loss

    if reduction == "mean":
        total_loss = np.mean(loss_per_sample)
    elif reduction == "sum":
        total_loss = np.sum(loss_per_sample)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}. Use 'mean' or 'sum'.")

    return float(total_loss)