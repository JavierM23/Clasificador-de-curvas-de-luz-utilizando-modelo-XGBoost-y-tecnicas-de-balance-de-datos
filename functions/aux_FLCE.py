import numpy as np
from numba import jit

np.random.seed(1994)

@jit(nopython=True)
def softmax(x):
    '''Softmax function with x as input vector.'''
    f = x
    f -=np.max(f)
    return np.exp(f) / np.sum(np.exp(f))



@jit(nopython=True)
def grad_hess(s, target, gamma, nclasses):
    eps = 1e-6
    st = 1-eps if s[int(target)] == 1 else s[int(target)]
    st_g = np.power(1-st, gamma) # (1-st)^gamma
    st_1 = np.power(1-st, -1) # (1-st)^(-1)
    st_g1 = np.prod(np.array([st_g, st_1])) # (1-st)^gamma * (1-st)^(-1) = (1-st)^(gamma-1)
    st_log = np.log(st) # log(st)

    g_array = np.zeros(nclasses)
    h_array = np.zeros(nclasses)

    g_t = np.prod(np.array([gamma, st_log, st_g1, st])) - st_g
    h_t = np.prod(np.array([gamma, st_g, st, np.prod(np.array([st_log, 1 - np.prod(np.array([gamma-1, st_1, st]))])) + 2]))

    for j in range(nclasses):
        assert target >= 0 or target <= nclasses
        if j == target:
            g = g_t - np.prod(np.array([st, g_t]))
            h1 = np.prod(np.array([h_t, 1-np.prod(np.array([st, st_1]))]))
        else:
            g = np.prod(np.array([-s[j], g_t]))
            h1 = 0

        h2 = np.prod(np.array([s[j], 1-s[j], g_t]))
        h3 = np.prod(np.array([s[j], s[j], h_t, st_1]))
        
        g_array[j] = g
        h_array[j] = h1 - h2 + h3

    return g_array, h_array