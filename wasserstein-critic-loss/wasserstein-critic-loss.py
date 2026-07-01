import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    real_scores = np.asarray(real_scores)
    fake_scores = np.asarray(fake_scores)
    
    mean_real_scores = np.mean(real_scores)
    mean_fake_scores = np.mean(fake_scores)

    loss = mean_fake_scores - mean_real_scores

    return float(loss)