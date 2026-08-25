import torch
import torch.nn.functional as F

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    """
    # first stage: BN1 --> ReLU --> Conv 1x1
    # prepare params
    x = torch.as_tensor(x, dtype=torch.float64)
    conv1_w = torch.as_tensor(conv1_weight, dtype=torch.float64)
    
    mean1 = torch.as_tensor(bn1_mean, dtype=torch.float64).view(1, -1, 1, 1)
    var1 = torch.as_tensor(bn1_var, dtype=torch.float64).view(1, -1, 1, 1)
    gamma1 = torch.as_tensor(bn1_gamma, dtype=torch.float64).view(1, -1, 1, 1)
    beta1 = torch.as_tensor(bn1_beta, dtype=torch.float64).view(1, -1, 1, 1)

    # BN1 (eval model) --> relu --> conv1x1 (padding=0)
    x_norm1 = (x - mean1) / torch.sqrt(var1 + eps)
    x_bn1 = gamma1 * x_norm1 + beta1
    x_relu1 = F.relu(x_bn1)
    y1 = F.conv2d(x_relu1, conv1_w, bias=None, stride=1, padding=0)

    # second stage: BN2 --> ReLU --> Conv 3x3
    # prepare params
    conv2_w = torch.as_tensor(conv2_weight, dtype=torch.float64)

    mean2 = torch.as_tensor(bn2_mean, dtype=torch.float64).view(1, -1, 1, 1)
    var2 = torch.as_tensor(bn2_var, dtype=torch.float64).view(1, -1, 1, 1)
    gamma2 = torch.as_tensor(bn2_gamma, dtype=torch.float64).view(1, -1, 1, 1)
    beta2 = torch.as_tensor(bn2_beta, dtype=torch.float64).view(1, -1, 1, 1)

    # BN2 (eval model) --> relu --> conv3x3 (padding=1 to keep H, W)
    y1_norm = (y1 - mean2) / torch.sqrt(var2 + eps)
    y1_bn = gamma2 * y1_norm + beta2
    y1_relu = F.relu(y1_bn)
    y2 = F.conv2d(y1_relu, conv2_w, bias=None, stride=1, padding=1)

    return y2