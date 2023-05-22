import numpy as np


def binary_find_windows(hist: np.ndarray, loss_func):
    res = hist.shape
    x = np.arange(0, res[1])
    y = np.arange(0, res[0])
    mat = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]).reshape((res[0], res[1], 2))
    loss_mat = loss_func(mat)
    loss_mat = loss_mat + hist
    return np.unravel_index(np.argmin(loss_mat, axis=None),  res)


def simple_loss_function(mat:np.ndarray,target_direction = None,x_weight=1,y_weight=1):
    if target_direction is None:
        target_direction = (mat.shape[0]/2,mat.shape[1]/2)
    return x_weight * np.abs(target_direction[0]-mat[:,:,1]) ** 2 + y_weight*np.abs(target_direction[1] -mat[:,:,0]) ** 2





