from collections import Counter
import math

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    c = len(candidate)
    r = len(reference)

    if c == 0 or r == 0:
        return 0.0

    precisions = []

    # calculate the modified precision p_n for each order from 1 to max_n
    for n in range(1, max_n + 1):
        if c < n:
            return 0.0

        # extract all n-grams using a sliding window
        cand_ngrams = [tuple(candidate[i:i + n]) for i in range(c - n + 1)]
        ref_ngrams = [tuple(reference[i:i + n]) for i in range(r - n + 1)]

        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)

        # calculate the clipped count and total candidate n-grams
        clipped_count = sum(min(count, ref_counts[ng]) for ng, count in cand_counts.items())
        total_cand_ngrams = len(cand_ngrams)

        if clipped_count == 0 or total_cand_ngrams == 0:
            return 0.0

        p_n = clipped_count / total_cand_ngrams
        precisions.append(p_n)

    # calculate BP
    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (r/c))

    # calculate the geometric mean of the precision values
    log_sum = sum(math.log(p) for p in precisions)
    geometric_mean = math.exp(log_sum/max_n)

    return float(bp * geometric_mean)