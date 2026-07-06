import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0:
        return 0.0

    TP = np.sum(y_true == y_pred)
    FP = FN = np.sum(y_true != y_pred)

    if (2*TP + FP + FN) == 0:
        return 0.0

    return float(2*TP/(2*TP + FP + FN))