import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    losses = []
    
    for y, p in zip(y_true, y_pred):
        # clipping p_hat into [eps, 1-eps]
        p_hat = max(eps, min(1.0 - eps, p))

        # Log Loss
        loss = -(y * math.log(p_hat) + (1.0 - y) * math.log(1.0 - p_hat))

        losses.append(float(loss))
    
    return losses