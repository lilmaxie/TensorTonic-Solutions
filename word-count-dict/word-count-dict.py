def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    word_counts = {}

    for sentence in sentences:
        for token in sentence:
            # if token existed: + 1, 0 if not exist
            word_counts[token] = word_counts.get(token, 0) + 1

    return word_counts