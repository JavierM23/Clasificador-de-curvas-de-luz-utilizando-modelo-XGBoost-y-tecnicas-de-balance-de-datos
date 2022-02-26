import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
import argparse

np.random.seed(1994)

def softmax(x):
    '''Softmax function with x as input vector.'''
    f = x
    f -=np.max(f)
    return np.exp(f) / np.sum(np.exp(f))

def grad_hess(s, target, gamma, nclasses):
    eps = 1e-6
    st = 1-eps if s[int(target)] == 1 else s[int(target)]
    st_g = np.power(1-st, gamma) # (1-st)^gamma
    st_1 = np.power(1-st, -1) # (1-st)^(-1)
    st_g1 = np.prod([st_g, st_1]) # (1-st)^gamma * (1-st)^(-1) = (1-st)^(gamma-1)
    st_log = np.log(st) # log(st)

    g_array = np.zeros(1,nclasses)
    h_array = np.zeros(1,nclasses)

    for j in range(nclasses):
        assert target >= 0 or target <= nclasses
        if j == target:
            g1 = np.prod([gamma, st_log, st_g1, st]) - st_g
            h1 = np.prod([gamma, st_g, st, np.prod([st_log, 1 - np.prod([gamma-1, st_1, st])]) + 2, 1-np.prod([st, st_1])])
        else:
            g1 = 0
            h1 = 0
        g2 = np.prod([s[j], st_g - np.prod([gamma, st_log, st_g1, st])])
        h2 = np.prod([s[j], 1-s[j], st_g - np.prod([gamma, st_log, st_g1, st])])
        h3 = np.prod([s[j], s[j], gamma, st_g1, st, np.prod([st_log, 1 - np.prod([gamma-1, st_1, st])]) + 2])
        
        g_array[j] = g1 + g2
        h_array[j] = h1 + h2 + h3

    return g_array, h_array


def softprob_obj(predt: np.ndarray, data: xgb.DMatrix):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.
    '''
    kRows = predt.shape[0]
    labels = data.get_label()
    kClasses = len(np.unique(labels))

    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            #h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            grad[r, c] = g
            hess[r, c] = h

    # Right now (XGBoost 1.0.0), reshaping is necessary
    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess




def focal_loss_obj2(predt: np.ndarray, data: xgb.DMatrix, gamma=0):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.
    '''
    kRows = predt.shape[0]
    labels = data.get_label()
    kClasses = len(np.unique(labels))

    if data.get_weight().size == 0:
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    for r in range(predt.shape[0]):
        target = labels[r]
        s = softmax(predt[r, :]) # prob from all classes
        st = 1-eps if s[int(target)] == 1 else s[int(target)]
        st_g = np.power(1-st, gamma)
        st_g1 = np.power(1-st, gamma-1)
        st_1 = np.power(1-st, -1)
        st_log = np.log(st+1e-9)
        for j in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            if j == target:
                g1 = np.prod([gamma, st_log, st_g1, st]) - st_g
                h1 = np.prod([st, gamma, st_g, st_log, 1 + np.prod([gamma-1, st, st_1]), 1 - np.prod([st, st_1])])
            else:
                g1 = 0
                h1 = 0
            g2 = np.prod([s[j], st_g - np.prod([gamma, st_log, st_g1, st])])
            h2 = np.prod([s[j], 1-s[j], st_g])
            h3 = np.prod([s[j], gamma, st_log, st_g1, st, np.prod([s[j], 1 - np.prod([gamma-1, st, st_1])]) - 1 + s[j]])

            g = g1 + g2
            g = g * weights[r]

            h = h1 + h2 + h3
            h = max((2.0 * h * weights[r]).item(), eps)

            grad[r, j] = g
            hess[r, j] = h

    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess

def focal_loss_obj1(predt: np.ndarray, data: xgb.DMatrix, gamma=0):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.
    '''
    kRows = predt.shape[0]
    labels = data.get_label()
    kClasses = len(np.unique(labels))

    if data.get_weight().size == 0:
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6

    for r in range(predt.shape[0]): #por cada instancia
        target = labels[r] #real class
        s = softmax(predt[r, :]) # prob from all classes
        g_array, h_array = grad_hess(s, target, gamma, predt.shape[1])
        
        g = g * weights[r]
        h = max((2.0 * h * weights[r]).item(), eps)

        for j in range(predt.shape[1]):
            grad[r, j] = g_array[j]
            hess[r, j] = h_array[j]

    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess





def focal_loss_obj(predt: np.ndarray, data: xgb.DMatrix):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.
    '''
    kRows = predt.shape[0]
    labels = data.get_label()
    kClasses = len(np.unique(labels))

    if data.get_weight().size == 0:
        weights = np.ones((kRows, 1), dtype=float)
    else:
        weights = data.get_weight()

    assert predt.shape == (kRows, kClasses)

    grad = np.zeros((kRows, kClasses), dtype=float)
    hess = np.zeros((kRows, kClasses), dtype=float)

    eps = 1e-6
    gamma = 0.5

    for r in range(predt.shape[0]):
        target = labels[r] #real class
        s = softmax(predt[r, :]) # prob from all classes
        st = 1-eps if s[int(target)] == 1 else s[int(target)]
        st_g = np.power(1-st, gamma) # (1-st)^gamma
        st_1 = np.power(1-st, -1) # (1-st)^(-1)
        st_g1 = np.prod([st_g, st_1]) # (1-st)^gamma * (1-st)^(-1) = (1-st)^(gamma-1)
        st_log = np.log(st) # log(st)

        for j in range(predt.shape[1]):
            assert target >= 0 or target <= kClasses
            if j == target:
                g1 = np.prod([gamma, st_log, st_g1, st]) - st_g
                h1 = np.prod([gamma, st_g, st, np.prod([st_log, 1 - np.prod([gamma-1, st_1, st])]) + 2, 1-np.prod([st, st_1])])
            else:
                g1 = 0
                h1 = 0
            g2 = np.prod([s[j], st_g - np.prod([gamma, st_log, st_g1, st])])
            h2 = np.prod([s[j], 1-s[j], st_g - np.prod([gamma, st_log, st_g1, st])])
            h3 = np.prod([s[j], s[j], gamma, st_g1, st, np.prod([st_log, 1 - np.prod([gamma-1, st_1, st])]) + 2])
        
            g = g1 + g2
            g = g * weights[r]

            h = h1 + h2 + h3
            h = max((2.0 * h * weights[r]).item(), eps)

            grad[r, j] = g
            hess[r, j] = h

    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess