import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    conv_w = torch.as_tensor(conv_weight, dtype=torch.float64)

    mean = torch.as_tensor(bn_mean, dtype=torch.float64).view(1, -1, 1, 1)
    var = torch.as_tensor(bn_var, dtype=torch.float64).view(1, -1, 1, 1)
    gamma = torch.as_tensor(bn_gamma, dtype=torch.float64).view(1, -1, 1, 1)
    beta = torch.as_tensor(bn_beta, dtype=torch.float64).view(1, -1, 1, 1)

    # BN (eval mode)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_bn = gamma * x_norm + beta

    # relu
    x_relu = F.relu(x_bn)

    # conv1x1 (compress from C -> C_out, padding=0, stride=1, no bias)
    x_conv = F.conv2d(x_relu, conv_w, stride=1, padding=0, bias=None)

    # dim reduction using 2x2 Average Pooling (stride=2) --> shape (N, C_out, H // 2, W // 2)
    out = F.avg_pool2d(x_conv, kernel_size=2, stride=2)

    return out