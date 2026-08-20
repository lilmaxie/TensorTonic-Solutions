import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    if not relevance_scores or k <= 0:
        return 0.0

    # if k is greater than the number of available elements, use the entire list.
    actual_k = min(k, len(relevance_scores))

    def compute_dcg(scores):
        dcg_val = 0.0
        for idx in range(actual_k):
            rel = scores[idx]
            # the 1-indexed position is (idx + 1) -> the denominator is log2(idx + 1 + 1) = log2(idx + 2).
            gain = (2.0**rel) - 1.0
            discount = math.log2(idx+2)
            dcg_val += gain/discount
        return dcg_val

    # actual DCG@k
    dcg = compute_dcg(relevance_scores)

    # construct an ideal list in descending order and calculate IDCG@k.
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = compute_dcg(ideal_scores)

    # handling the case where IDCG is zero (all relevance scores are zero)
    if idcg == 0:
        return 0.0

    return float(dcg/idcg)