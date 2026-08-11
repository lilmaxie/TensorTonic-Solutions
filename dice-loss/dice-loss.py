import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    p_flat = p.ravel()
    y_flat = y.ravel()

    intersection = np.sum(p_flat * y_flat)
    sum_p = np.sum(p_flat)
    sum_y = np.sum(y_flat)

    dice_coef = (2*intersection + eps) / (sum_p + sum_y + eps)
    
    return float(1-dice_coef)