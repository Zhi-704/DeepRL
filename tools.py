import numpy as np

def GAE(deltas, gamma, lam):
    length = len(deltas)
    output = np.zeros(length)
    for i in range(length-1, -1, -1):
        if i == length-1:
            output[i] = deltas[i]
        else:
            output[i] = deltas[i] + gamma*lam * output[i+1]
    return output