import math

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    K = len(predictions)

    q = []
    for i in range(K):
        if i == target:
            q_i = (1-epsilon) + (epsilon/K)
        else:
            q_i = epsilon/K
        q.append(q_i)

    loss = 0.0
    for q_i, p_i in zip(q, predictions):
        loss -= q_i*math.log(p_i)

    return float(loss)