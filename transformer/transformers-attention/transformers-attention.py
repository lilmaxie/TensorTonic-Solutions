import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # take the dimension d_k from the last dimension of the Key tensor
    d_k = K.size(-1)
    # Calculate the raw score matrix: S = (Q @ K.T) / sqrt(d_k)
    # K.transpose(-2, -1) swaps the last two dimensions from (..., seq_len_k, d_k) to (..., d_k, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # calculate attention weight distribution via the Softmax function along the Key dimension (dim=-1)
    attention_weights = F.softmax(scores, dim=-1)
    # multiply the weight by the Value to get the context vector: Output = Attention_Weights @ V
    output = torch.matmul(attention_weights, V)

    return output