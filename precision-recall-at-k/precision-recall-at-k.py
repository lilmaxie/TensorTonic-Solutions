def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]

    relevant_set = set(relevant)

    # count the hits (intersection between top_k and relevant)
    hits = sum(1 for item in top_k if item in relevant_set)

    precision = hits/k
    recall = hits/len(relevant)
    
    return [float(precision), float(recall)]