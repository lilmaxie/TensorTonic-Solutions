import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    # create a copy to avoid directly modifying the original matrix
    W_in_updated = W_in.clone()
    W_out_updated = W_out.clone()

    # snapshot the entire vector before the update
    v_c = W_in[center_id].clone()
    u_o = W_out[pos_id].clone()

    neg_id_list = neg_ids.tolist() if isinstance(neg_ids, torch.Tensor) else list(neg_ids)
    u_negs = [W_out[idx].clone() for idx in neg_id_list]

    # calculate Error Coefficients
    # positive: sigma(score_o) - 1
    score_o = torch.dot(v_c, u_o)
    coeff_pos = torch.sigmoid(score_o) - 1.0

    # negative: sigma(score_ni)
    coeff_negs = []
    for u_n in u_negs:
        score_n = torch.dot(v_c, u_n)
        coeff_negs.append(torch.sigmoid(score_n))

    # calculate the gradient and update W_in (center_id)
    # grad_v_c = (sigma(s_o) - 1) * u_o + sum_i(sigma(s_ni) * u_ni)
    grad_v_c = coeff_pos * u_o
    for coeff_n, u_n in zip(coeff_negs, u_negs):
        grad_v_c += coeff_n * u_n

    W_in_updated[center_id] -= lr * grad_v_c

    # calculate gradient and update W_out (pos_id & neg_ids)
    # updated based on the correct context: grad_u_o = (sigma(s_o) - 1) * v_c
    grad_u_o = coeff_pos * v_c
    W_out_updated[pos_id] -= lr * grad_u_o

    # update negative terms: grad_u_ni = sigma(s_ni) * v_c (automatically accumulated if IDs match)
    for coeff_n, n_id in zip(coeff_negs, neg_id_list):
        grad_u_n = coeff_n * v_c
        W_out_updated[n_id] -= lr * grad_u_n

    return W_in_updated, W_out_updated