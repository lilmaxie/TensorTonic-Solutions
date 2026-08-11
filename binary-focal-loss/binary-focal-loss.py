import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    
    # Tính p_t trên mảng NumPy
    p_t = np.where(y == 1, p, 1.0 - p)
    
    # Tính Focal Loss
    loss = -alpha * ((1.0 - p_t) ** gamma) * np.log(p_t)
    
    return float(np.mean(loss))