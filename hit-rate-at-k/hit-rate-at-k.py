def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    n_users = len(recommendations)

    if n_users == 0:
        return 0.0

    hits = 0

    for recs, gt in zip(recommendations, ground_truth):
        top_k = recs[:k]

        top_k_set = set(top_k)
        gt_set = set(gt)

        if top_k_set & gt_set:
            hits+=1

    return float(hits/n_users)