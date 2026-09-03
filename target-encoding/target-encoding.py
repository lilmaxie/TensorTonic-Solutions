def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    category_sums = {}
    category_counts = {}

    # use a single pass over the data to calculate the total target and the freq
    for cat, target in zip(categories, targets):
        if cat not in category_sums:
            category_sums[cat] = float(target)
            category_counts[cat] = 1
        else:
            category_sums[cat] += float(target)
            category_counts[cat] += 1

    # calculate the average value for each unique category
    category_means = {
        cat: category_sums[cat] / category_counts[cat]
        for cat in category_sums
    }

    # replace each original category with the calculated mean value
    encoded_categories = [category_means[cat] for cat in categories]

    return encoded_categories