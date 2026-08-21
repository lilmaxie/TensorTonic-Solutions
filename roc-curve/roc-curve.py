import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    # sort the data in descending order of y_score
    desc_order = np.argsort(-y_score)
    y_true_sorted = y_true[desc_order]
    y_score_sorted = y_score[desc_order]

    # calculate the cumulative number of True Positives and False Positives
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1.0 - y_true_sorted)

    # identify unique threshold positions (Handling Tied Scores)
    # get the indices where the score value changes, plus the last element
    distinct_indices = np.where(np.diff(y_score_sorted) != 0)[0]
    threshold_indices = np.r_[distinct_indices, y_true_sorted.size - 1]

    # extract TP, FP, and Threshold values ​​at the actual transition points
    tps = tps[threshold_indices]
    fps = fps[threshold_indices]
    thresholds = y_score_sorted[threshold_indices]

    # add the initial starting point: FPR=0, TPR=0 corresponding to threshold = +inf
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    thresholds = np.r_[np.inf, thresholds]

    # calculate normalized TPR and FPR
    total_pos = tps[-1]
    total_neg = fps[-1]

    tpr = tps / total_pos if total_pos > 0 else np.zeros_like(tps)
    fpr = fps / total_neg if total_neg > 0 else np.zeros_like(fps)

    return fpr, tpr, thresholds