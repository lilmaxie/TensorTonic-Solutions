def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    T = len(predictions)
    N = len(predictions[0])

    final_labels = []

    # iterate through each data sample (corresponding to each column)
    for i in range(N):
        counts = {}
        # count the votes from all T trees for sample i
        for t in range(T):
            label = predictions[t][i]
            counts[label] = counts.get(label, 0) + 1

        # label with the highest number of votes; if tie, choose the smallest label
        # comparison key: (number of votes, -label value)
        best_label = max(counts.keys(), key=lambda l: (counts[l], -l))
        final_labels.append(int(best_label))

    return final_labels