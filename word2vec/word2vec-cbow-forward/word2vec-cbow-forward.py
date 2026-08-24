import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int, W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the CBOW cross-entropy loss for predicting target_id from the averaged context.
    """
    # look up the embeddings of contextual words and calculate the average vector h
    # W_in[context_ids] has shape (num_context_words, D) -> mean(dim=0) gives shape (D,)
    context_vectors = W_in[context_ids]
    h = context_vectors.mean(dim=0)

    # score the entire dictionary (Logits)
    # W_out @ h: (vocab_size, D) x (D,) -> shape (vocab_size,)
    logits = torch.matmul(W_out, h)

    # calculate Negative Log-Likelihood Loss using F.log_softmax
    # F.log_softmax xử lý an toàn số học bằng LogSumExp trick
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs[target_id]

    return loss