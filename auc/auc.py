import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)    

    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have the same length")
    if len(fpr) < 2:
        raise ValueError("At least 2 points are required to compute AUC")

    if hasattr(np, 'trapezoid'):
        result = np.trapezoid(tpr, fpr)
    else:
        result = np.trapz(tpr, fpr)

    return float(result)