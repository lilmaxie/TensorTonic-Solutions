import numpy as np

def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # flatten image into 1D array
    flat_img = np.asarray(image).ravel()

    # count freq of the values, guarantee length 256
    hist = np.bincount(flat_img, minlength=256)

    return hist.tolist()