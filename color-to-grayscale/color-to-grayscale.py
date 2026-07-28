import numpy as np

def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # ITU-R BT.601 standard weight
    weights = np.asarray([0.299, 0.587, 0.114])

    grayscale = np.dot(image, weights)

    return grayscale.tolist()