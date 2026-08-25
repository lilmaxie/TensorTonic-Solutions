import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    channel_history = [stem_channels]
    current_channel = stem_channels
    num_blocks = len(block_layers)

    for idx, n in enumerate(block_layers):
        # channels number after passing a dense block with n layers: C_block = C_in + n * k
        c_block = current_channel + n * growth_rate
        channel_history.append(c_block)

        # channel-compressing Transition Layer (only for non-final blocks)
        if idx < num_blocks-1:
            c_trans = int(math.floor(c_block*compression))
            channel_history.append(c_trans)

            current_channel = c_trans

    return torch.tensor(channel_history, dtype=torch.int64)