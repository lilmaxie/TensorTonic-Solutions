import torch

def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,), a probability distribution that sums to 1.
    """
    counts_tensor = torch.as_tensor(counts, dtype=torch.float64)

    powered_counts = counts_tensor ** alpha
    prob_dist = powered_counts/torch.sum(powered_counts)

    return prob_dist
