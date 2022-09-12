import numpy as np
import distrax

def kld_normal(y):
    unique, counts = np.unique(y, return_counts=True)
    full_counts = np.zeros(y.max() - y.min() + 1)
    full_counts[unique - y.min()] = counts
    probs = full_counts / len(y)
    dist = distrax.Normal(y.mean(), y.std())
    qdist = distrax.Quantized(dist, int(y.min()), int(y.max()))
    gprobs = qdist.prob(np.arange(y.min(), y.max()+1))
    indices1 = np.where(probs)
    indices2 = np.where(gprobs)
    indices = np.intersect1d(indices1, indices2)
    probs1 = probs[indices]
    gprobs1 = gprobs[indices]
    return np.sum(np.log2(probs1/gprobs1) * probs1)
    
