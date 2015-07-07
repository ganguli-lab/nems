"""
Tools for visualizing models

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from toolz import curry, compose
from scipy.stats import skew


@curry
def contour(W, n=3, **kwargs):
    """
    Plots a contour of a spatiotemporal filter onto the current figure

    Parameters
    ----------
    W : array_like
        A matrix consisting of a spatiotemporal filter

    n : Integer (optional)
        The number of contour levels to plot (default: 3)

    \\*\\*kwargs : keyword arguments
        Any keyword arguments for matplotlib.pyplot.contour

    """

    # type checks
    assert type(W) == np.ndarray, "The argument must be a numpy array"
    assert W.ndim == 2, "The argument must be a matrix (two dimensions)"

    # skew-positive
    W *= np.sign(skew(W.ravel()))

    # plot the contour with the given keyword arguments
    plt.contour(W, n, **kwargs)


@curry
def plot(mdl, cmap='seismic'):
    """
    Plots the parameters of a NeuralEncodingModel

    """
    nsub = mdl.num_subunits

    Phi = mdl.tentparams['Phi']
    tx = mdl.tentparams['tent_x']

    # create the figure
    fig = plt.figure(figsize=(12, 8))

    maxval = compose(np.max, np.abs, np.vstack)(mdl.theta['W'])

    # plot the filters
    for j in range(nsub):
        W = mdl.theta['W'][j]
        ax = fig.add_subplot(2, nsub, j+1)
        ax.pcolor(W, cmap=cm.__dict__[cmap], vmin=-maxval, vmax=maxval)
        ax.set_xticks([])
        ax.set_yticks([])

    # plot the nonlinearities
    for j in range(nsub):
        f = mdl.theta['f'][j]
        ax = fig.add_subplot(2, nsub, j+nsub+1)
        ax.plot(tx, Phi.dot(f[:-1]) + f[-1], 'k-')
        ax.set_xlim(-4, 4)


@curry
def sort(W):
    """
    Sorts the given list of spatiotemporal filters by the locataion
    of the spatial peak
    """

    spatial = lambda w: np.linalg.svd(w)[0][:, 0]
    maxidx = lambda v: np.argmax(v * np.sign(skew(v)))
    return sorted(W, key=compose(maxidx, spatial))


@curry
def psub(idx, wi, nrows=1, nsub=4, offset=1, v=0.2):
    plt.subplot(nrows, nsub, idx+offset)
    plt.pcolor(wi, cmap='seismic', vmin=-v, vmax=v)
    plt.xticks([])
    plt.yticks([])
