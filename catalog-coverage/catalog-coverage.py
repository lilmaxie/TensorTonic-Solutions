def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    unique_recommend_items = set()

    for user_recs in recommendations:
        unique_recommend_items.update(user_recs)

    if n_items == 0:
        return 0.0

    coverage = len(unique_recommend_items) / n_items

    return float(coverage)