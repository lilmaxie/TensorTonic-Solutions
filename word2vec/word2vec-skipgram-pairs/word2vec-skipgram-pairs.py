import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    n = len(token_ids)

    if n <= 1 or window <= 0:
        return torch.zeros((0,2), dtype=torch.int64)

    # convert to python list get faster loop
    tokens = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
    pairs = []

    # go through each position starting from the center position (Center Position) i
    for i in range(n):
        center_token = tokens[i]

        # determine the context window range [start, end)
        start = max(0, i-window)
        end = min(n, i+window+1)

        # iterate through the context positions j (Context Position)
        for j in range(start, end):
            if i != j:
                pairs.append([center_token, tokens[j]])

    # returns an empty tensor with the correct shape (0, 2) if there are no pairs
    if len(pairs) == 0:
        return torch.zeros((0, 2), dtype=torch.int64)

    return torch.tensor(pairs, dtype=torch.int64)