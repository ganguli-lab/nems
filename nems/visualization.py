"""
Tools for visualizing models
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
from jetpack import img
import pyret.visualizations as viz
from toolz import compose, curry
import pyret.filtertools as ft


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

    # skew-positive
    W *= np.sign(skew(W.ravel()))

    # plot the contour with the given keyword arguments
    plt.contour(-W, n, **kwargs)


def plotcells(W, n=15, alpha=0.4, palette='pastel'):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # colors = palettable.colorbrewer.qualitative.Set1_9.colors

    for wi in W:

        spf = np.linalg.eigh(np.cov(wi.T.reshape(-1, n * n).T))[1][:, -1].reshape(n, n)
        tx = np.arange(n)
        ell = ft.fit_ellipse(tx, tx, spf)
        viz.ellipse(ell, ax=ax)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        # ell.set_facecolor(color)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    return ax


def hcat(filters, newshape=None, figsize=(8, 5)):
    """
    horizontally concatenate a bunch of matrices and display the result as an image
    """
    if newshape is None:
        newshape = filters[0].shape

    fig = plt.figure(figsize=figsize)
    img(np.hstack([f.reshape(newshape) for f in filters]), fig=fig)


@curry
def plot(mdl, cmap='seismic'):
    """
    Plots the parameters of a NeuralEncodingModel

    """
    nsub = mdl.num_subunits

    # create the figure
    fig = plt.figure(figsize=(4 * nsub, 8))

    maxval = compose(np.max, np.abs, np.vstack)(mdl.theta['W'])

    # plot the filters
    for j in range(nsub):
        W = mdl.theta['W'][j]
        ax = fig.add_subplot(2, nsub, j + 1)
        ax.pcolor(W, cmap=cm.__dict__[cmap], vmin=-maxval, vmax=maxval)
        ax.set_xticks([])
        ax.set_yticks([])

    # plot the nonlinearities
    for j in range(nsub):
        f = mdl.theta['f'][j]
        ax = fig.add_subplot(2, nsub, j + nsub + 1)
        mdl.tents.plot(f)
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
    plt.subplot(nrows, nsub, idx + offset)
    plt.pcolor(wi, cmap='seismic', vmin=-v, vmax=v)
    plt.xticks([])
    plt.yticks([])
