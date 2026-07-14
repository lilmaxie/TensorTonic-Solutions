import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)

    if len(rater1) != len(rater2):
        raise ValueError("Two raters must have the same size")

    n = len(rater1)
    if n == 0:
        return 0.0
        
    po = np.sum(rater1==rater2)/n
    
    unique_classes = np.unique(np.concatenate([rater1, rater2]))
    pe = 0.0
    for k in unique_classes:
        count1 = np.sum(rater1==k)
        count2 = np.sum(rater2==k)

        pe += (count1/n) * (count2/n)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    K = (po-pe)/(1.0-pe)
    
    return float(K)