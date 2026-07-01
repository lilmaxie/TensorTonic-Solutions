import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.asarray(x)

    act_fnc = 1/(1+np.exp(-x))
    swish = x*act_fnc

    return swish