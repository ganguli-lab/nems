"""
Tools for visualizing models
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew


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


def plot(mdl, cmap='seismic', axis=0):
    """
    Plots the parameters of a NeuralEncodingModel

    """
    nsub = mdl.num_subunits

    # create the figure
    fig = plt.figure(figsize=(4 * nsub, 8))

    # Used to define the colormap range.
    maxval = np.max(np.abs(np.vstack(mdl.theta['W'])))

    # Plot the filters.
    for j in range(nsub):
        W = mdl.theta['W'][j]

        # Reshape filter if three dimensional.
        if len(mdl.filter_dims) > 2:
            W = W.reshape(mdl.filter_dims)
            assert axis == 0 or axis == 1, "Invalid spatial axis"

            # Get maximum intensity value from the other spatial axis.
            W = np.rollaxis(W, 1-axis)
            maxi = np.unravel_index(np.abs(W).argmax(),
                                    mdl.filter_dims)[0]

            # Discard the other spatial axis.
            W = W[maxi]

        ax = fig.add_subplot(2, nsub, j + 1)
        ax.pcolor(W, cmap=cm.__dict__[cmap], vmin=-maxval, vmax=maxval)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot the nonlinearities.
    for j in range(nsub):
        f = mdl.theta['f'][j]
        ax = fig.add_subplot(2, nsub, j + nsub + 1)
        mdl.tents.plot(f)
        ax.set_xlim(-4, 4)
