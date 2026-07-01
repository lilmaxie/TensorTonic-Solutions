import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    losses = np.maximum(0, margin - y_true * y_score)

    if reduction == "mean":
        loss = np.mean(losses)
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
        
    return float(loss)
    