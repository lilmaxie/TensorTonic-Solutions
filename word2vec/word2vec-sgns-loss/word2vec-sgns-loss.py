import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    # positive dot product
    pos_score = torch.dot(center_vec, pos_vec)

    # calculate the scalar product of k and the negative vector using matrix-vector multiplication.
    # neg_vecs @ center_vec --> shape(k,)
    neg_scores = torch.matmul(neg_vecs, center_vec)

    # calculate the loss using the softplus function to eliminate numerical overflow
    # -log(sigmoid(pos_score)) = softplus(-pos_score)
    pos_loss = F.softplus(-pos_score)
    # -log(sigmoid(-neg_scores)) = softplus(+neg_scores)
    neg_loss = torch.sum(F.softplus(neg_scores))

    total_loss = pos_loss + neg_loss

    return total_loss