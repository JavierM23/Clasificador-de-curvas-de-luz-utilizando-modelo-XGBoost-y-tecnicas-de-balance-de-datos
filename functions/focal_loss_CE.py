import numpy as np
import xgboost as xgb
from aux_FLCE import softmax, grad_hess


def focal_loss_obj(predt: np.ndarray, data: xgb.DMatrix, gamma=0):
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

        for j in range(predt.shape[1]):
            grad[r, j] = g_array[j] * weights[r]
            hess[r, j] = max((2.0 * h_array[j] * weights[r]).item(), eps)

    grad = grad.reshape((kRows * kClasses, 1))
    hess = hess.reshape((kRows * kClasses, 1))
    return grad, hess