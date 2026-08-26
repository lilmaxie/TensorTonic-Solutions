import numpy as np

def make_vgg_config(variant: str) -> list:
    """
    Return the layer configuration for a VGG variant.
    """
    variant = variant.lower()
    if variant == 'vgg11':
        return [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    elif variant == 'vgg13':
        return [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    elif variant == 'vgg16':
        return [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    elif variant == 'vgg19':
        return [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    else:
        raise ValueError(f"Unknown variant: {variant}")