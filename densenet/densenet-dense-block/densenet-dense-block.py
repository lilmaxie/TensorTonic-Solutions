import torch

def dense_block(x, layers, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, C + L*growth_rate, H, W).
    """
    x = torch.as_tensor(x, dtype=torch.float64)

    # save feature maps (start with x_0)
    feats = [x]

    # iterate sequentially through each Composite Layer in the block
    for layer in layers:
        # concatenate all previous feature maps along the channel dimension (dim=1)
        # shape (N, C+l*growth_rate, H, W)
        concat_input = torch.cat(feats, dim=1)

        # prepare params into (1, C_accumulated, 1, 1) to broadcast
        gamma = torch.as_tensor(layer['bn_gamma'], dtype=torch.float64).view(1, -1, 1, 1)
        beta = torch.as_tensor(layer['bn_beta'], dtype=torch.float64).view(1, -1, 1, 1)
        mean = torch.as_tensor(layer['bn_mean'], dtype=torch.float64).view(1, -1, 1, 1)
        var = torch.as_tensor(layer['bn_var'], dtype=torch.float64).view(1, -1, 1, 1)
        conv_w = torch.as_tensor(layer['conv_weight'], dtype=torch.float64)

        # BN (eval mode)
        x_norm = (concat_input - mean) / torch.sqrt(var + eps)
        x_bn = gamma *  x_norm + beta

        # relu
        x_relu = F.relu(x_bn)

        # conv 3x3, padding=1, stride=1, no bias --> shape (N, growth_rate, H, W)
        new_feats = F.conv2d(x_relu, conv_w, stride=1, padding=1)

        feats.append(new_feats)

    # concatenation of x_0 and the entire set of new L features --> shape (N, C + L * growth_rate, H, W)
    return torch.cat(feats, dim=1)