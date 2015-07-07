"""
Utilities and helper functions for the neural encoding models package

"""

# imports
import numpy as np
from toolz import curry, compose

# exports
__all__ = ['rolling_window', 'nrm']


def rolling_window(a, window):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    a : array_like
        Array to add rolling window to
    window : int
        Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    """
    assert window >= 1, "`window` must be at least 1."
    assert window < a.shape[-1], "`window` is too long."

    # # with strides
    shape = a.shape[:-1] + (a.shape[-1] - window, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def nrm(x):
    """
    Normalizes data in the given array x by the (vectorized) 2-norm

    Parameters
    ----------
    x : array_like
        The input to be normalized

    Returns
    -------
    xn : array_like
        A version of the input array that has been scaled so it has a unit vectorized 2-norm

    """
    return x / np.linalg.norm(x.ravel())


@curry
def microshift(dx, W):
    return W + dx*np.vstack(map(np.gradient, W.T)).T


@curry
def shift(i, W):
    return np.vstack((W[i:, :], W[:i, :]))


arr = compose(np.array, list)
