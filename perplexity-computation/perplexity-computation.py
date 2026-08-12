def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    total_log_prob = 0.0
    num_token = len(actual_tokens)

    for i, token_index in enumerate(actual_tokens):
        prob = prob_distributions[i][token_index]
        total_log_prob += math.log(prob)

    cross_entropy = -total_log_prob / num_token

    return math.exp(cross_entropy)