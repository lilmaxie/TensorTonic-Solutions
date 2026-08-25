import torch
import torch.nn.functional as F


def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-3x3Conv (padding 1, no bias) producing growth_rate channels.
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    conv_w = torch.as_tensor(conv_weight, dtype=torch.float64)

    mean = torch.as_tensor(bn_mean, dtype=torch.float64).view(1, -1, 1, 1)
    var = torch.as_tensor(bn_var, dtype=torch.float64).view(1, -1, 1, 1)
    gamma = torch.as_tensor(bn_gamma, dtype=torch.float64).view(1, -1, 1, 1)
    beta = torch.as_tensor(bn_beta, dtype=torch.float64).view(1, -1, 1, 1)

    # BatchNorm (EVAL mode)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_bn = gamma * x_norm + beta

    # ReLU 
    x_relu = F.relu(x_bn)

    # conv 3x3 (padding=1, stride=1, no bias)
    out = F.conv2d(x_relu, conv_w, bias=None, stride=1, padding=1)

    return out


def dense_block(x, layers, eps):
    """
    Returns torch.Tensor: concat of x and every composite-layer output (channels grow by growth_rate per layer).
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    feats = [x]

    for layer in layers:
        # concat all previously accumulated feature maps
        concat_input = torch.cat(feats, dim=1)

        # new characteristics achieved through composite layers
        new_feat = composite_layer(
            concat_input,
            layer['bn_gamma'],
            layer['bn_beta'],
            layer['bn_mean'],
            layer['bn_var'],
            layer['conv_weight'],
            eps
        )
        feats.append(new_feat)

    return torch.cat(feats, dim=1)


def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-1x1Conv then 2x2 average pool with stride 2 (channels compressed, H and W halved).
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    conv_w = torch.as_tensor(conv_weight, dtype=torch.float64)

    mean = torch.as_tensor(bn_mean, dtype=torch.float64).view(1, -1, 1, 1)
    var = torch.as_tensor(bn_var, dtype=torch.float64).view(1, -1, 1, 1)
    gamma = torch.as_tensor(bn_gamma, dtype=torch.float64).view(1, -1, 1, 1)
    beta = torch.as_tensor(bn_beta, dtype=torch.float64).view(1, -1, 1, 1)

    # BatchNorm (EVAL mode)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_bn = gamma * x_norm + beta

    # ReLU Activation
    x_relu = F.relu(x_bn)

    # conv 1x1 (padding=0, stride=1, no bias)
    x_conv = F.conv2d(x_relu, conv_w, bias=None, stride=1, padding=0)

    # avgpool 2x2 (stride=2)
    out = F.avg_pool2d(x_conv, kernel_size=2, stride=2)

    return out


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, num_classes) with class logits.
    """
    x = torch.as_tensor(x, dtype=torch.float64)

    # stem Convolution 3x3 (padding=1, stride=1, no bias)
    stem_conv = torch.as_tensor(weights.get('stem_conv', weights.get('stem')), dtype=torch.float64)
    h = F.conv2d(x, stem_conv, bias=None, stride=1, padding=1)

    # iterate through the alternating Dense Blocks and Transition Layers.
    blocks = weights['blocks']
    transitions = weights.get('transitions', [])
    num_blocks = len(blocks)

    for i in range(num_blocks):
        # pass through Dense Block [i]
        h = dense_block(h, blocks[i], eps)

        # apply a Transition Layer to all blocks except the last one.
        if i < num_blocks - 1:
            trans = transitions[i]
            h = transition_layer(
                h,
                trans['bn_gamma'],
                trans['bn_beta'],
                trans['bn_mean'],
                trans['bn_var'],
                trans['conv_weight'],
                eps
            )

    # Final Batch Normalization & ReLU
    if 'final_bn' in weights:
        final_bn = weights['final_bn']
        f_gamma = final_bn['bn_gamma']
        f_beta = final_bn['bn_beta']
        f_mean = final_bn['bn_mean']
        f_var = final_bn['bn_var']
    else:
        f_gamma = weights['final_bn_gamma']
        f_beta = weights['final_bn_beta']
        f_mean = weights['final_bn_mean']
        f_var = weights['final_bn_var']

    mean_f = torch.as_tensor(f_mean, dtype=torch.float64).view(1, -1, 1, 1)
    var_f = torch.as_tensor(f_var, dtype=torch.float64).view(1, -1, 1, 1)
    gamma_f = torch.as_tensor(f_gamma, dtype=torch.float64).view(1, -1, 1, 1)
    beta_f = torch.as_tensor(f_beta, dtype=torch.float64).view(1, -1, 1, 1)

    z = gamma_f * ((h - mean_f) / torch.sqrt(var_f + eps)) + beta_f
    z = F.relu(z)

    # Global Average Pooling (GAP) across the spatial dimensions (H, W) --> shape (N, C_final, H, W) -> (N, C_final)
    pooled = z.mean(dim=(2, 3))

    # FC Classifier (Logits)
    fc_w = torch.as_tensor(weights.get('fc_weight', weights.get('fc', {}).get('weight')), dtype=torch.float64)
    fc_b = weights.get('fc_bias', weights.get('fc', {}).get('bias', None))

    if fc_b is not None:
        fc_b = torch.as_tensor(fc_b, dtype=torch.float64)
        logits = F.linear(pooled, fc_w, fc_b)
    else:
        logits = F.linear(pooled, fc_w, bias=None)

    return logits