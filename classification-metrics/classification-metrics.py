import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true)

    if N == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    accuracy = float(np.mean(y_true == y_pred))

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return {"accuracy": accuracy, "precision": p, "recall": r, "f1": f1}

    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == "micro":
        total_tp = sum(np.sum((y_true == c) & (y_pred == c)) for c in classes)
        total_fp = sum(np.sum((y_true != c) & (y_pred == c)) for c in classes)
        total_fn = sum(np.sum((y_true == c) & (y_pred != c)) for c in classes)

        p = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        r = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return {"accuracy": accuracy, "precision": p, "recall": r, "f1": f1}

    p_list, r_list, f1_list, support_list = [], [], [], []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)

        p_c = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r_c = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_c = float(2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0

        p_list.append(p_c)
        r_list.append(r_c)
        f1_list.append(f1_c)
        support_list.append(support)

    if average == "macro":
        precision = float(np.mean(p_list))
        recall = float(np.mean(r_list))
        f1 = float(np.mean(f1_list))
    elif average == "weighted":
        total_support = sum(support_list)
        if total_support > 0:
            weights = np.array(support_list) / total_support
            precision = float(np.sum(np.array(p_list) * weights))
            recall = float(np.sum(np.array(r_list) * weights))
            f1 = float(np.sum(np.array(f1_list) * weights))
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0
    else:
        raise ValueError(f"Unknown average strategy: {average}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }