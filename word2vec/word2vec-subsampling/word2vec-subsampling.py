import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # convert `counts` to float to perform floating-point division
    counts_float = counts.to(dtype=torch.float32)

    # calculate the total number of words N and the relative frequency f(w) = count(w) / N
    total_count = torch.sum(counts_float)
    freq = counts_float/total_count

    # calculate the ratio sqrt(t / f(w))
    raw_probs = torch.sqrt(t/freq)

    # the upper limit is 1.0 (rare words with f(w) <= t will have a probability of 1.0)
    keep_probs = torch.clamp(raw_probs, max=1.0)
    
    return keep_probs