from collections import Counter

def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    if not tokens:
        return {}, {}

    # build vocab V include unique word
    vocab = set(tokens)
    V_size = len(vocab)

    # count freq of all adjacent bigram (tokens[i] and tokens[i+1])
    counts = {}
    context_counts = {w: 0 for w in vocab} # all bigram start with w

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        counts[bigram] = counts.get(bigram, 0) + 1
        context_counts[tokens[i]] += 1

    # calculate prob smoothen P(w2|w1) for all space VxV
    probs = {}
    for w1 in vocab:
        denominator = context_counts[w1] + V_size

        for w2 in vocab:
            numerator = counts.get((w1, w2), 0) + 1
            probs[(w1, w2)] = numerator/denominator

    return counts, probs