import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def GAE(deltas, gamma, lam):
    length = len(deltas)
    output = np.zeros(length)
    for i in range(length-1, -1, -1):
        if i == length-1:
            output[i] = deltas[i]
        else:
            output[i] = deltas[i] + gamma*lam * output[i+1]
    return output


@nb.njit(fastmath=True)
def standardize(values):
    """Standardize by subtracting the mean and dividing by the standard deviation."""
    mean = np.mean(values)
    std = np.std(values)
    return (values - mean) / std

def clip(value, ϵ):
    """Clip the value within [1 - ϵ, 1 + ϵ]"""
    if value < (1 - ϵ):
        return (1 - ϵ)
    if value > (1 + ϵ):
        return (1 + ϵ)
    else:
        return value