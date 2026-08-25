import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    conv_weight = torch.as_tensor(conv_weight, dtype=torch.float64)
    
    # reshape 1D vectors (C,) into 4D (1, C, 1, 1) to broadcast across N, H, and W.
    mean = torch.as_tensor(bn_mean, dtype=torch.float64).view(1, -1, 1, 1)
    var = torch.as_tensor(bn_var, dtype=torch.float64).view(1, -1, 1, 1)
    gamma = torch.as_tensor(bn_gamma, dtype=torch.float64).view(1, -1, 1, 1)
    beta = torch.as_tensor(bn_beta, dtype=torch.float64).view(1, -1, 1, 1)

    # batch norm (eval mode)
    x_norm = (x-mean) / torch.sqrt(var+eps)
    x_bn = gamma * x_norm + beta

    # activation func 
    x_relu = F.relu(x_bn)

    # conv with same padding and no bias
    # output shape: (N, growth_rate, H, W)
    out = F.conv2d(x_relu, conv_weight, bias=None, stride=1, padding=1)

    return out