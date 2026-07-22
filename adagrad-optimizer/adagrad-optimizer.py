import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    w = np.asarray(w)
    g = np.asarray(g)
    G = np.asarray(G)

    G_next = G + g**2

    w_next = w - lr*g/(np.sqrt(G_next + eps))

    return w_next.tolist(), G_next.tolist()