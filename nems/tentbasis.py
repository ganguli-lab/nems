"""
Functions for generating and evaluating functions and gradients
composed of linear combinations of tent basis functions.

This module allows you to:

- Generate tent functions spanning an input range (either Gaussian or Linear tents)
- Generate a matrix where the ith column is the value of either the tent function
    :math:`\phi(x-i)` or its derivative :math:`\phi'(x-i)` for an input vector x.

Notes
-----
These have the following functional form:

.. math:: f(x) = \sum_i w_i \phi(x-i)

where each tent function :math:`\phi` is a _fixed_ basis function, shifted
along the input dimension to tile the input space. The function :math:`f(x)`
is parameterized as a linear combination (with weights given by :math:`\alpha`)
of these tent functions.

"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Linear', 'Gaussian', 'Ispline', 'Nonlinearity', 'make_rcos_basis']


class Nonlinearity(object):

    def __init__(self, tent_span, num_tents):
        """
        Initialize a nonlinearity parameterized by tent basis functions

        Arguments
        ---------
        num_tent_samples    -- number of samples for the input range (tent_x)
        tent_span           -- tuple designating the input range e.g. (min, max)
        num_tents           -- number of tent basis functions to use
        tent_type           -- shape of the basis functions (defaults to 'gaussian'. other option: 'linear')

        """

        self.tent_span = tent_span
        self.num_tents = num_tents
        self.num_params = num_tents + 1
        self.centers = np.linspace(tent_span[0], tent_span[1], num_tents)

    def __str__(self):
        """
        TODO: add documentation
        """
        name = str(self.__class__).rpartition('.')[2].partition("'")[0]
        return "{} nonlinearity with {} parameters.".format(name, self.num_params)

    def fit(self, x, y):
        """
        Estimate parameters (using a least squares fit) to approximate f(x) = y
        """

        Phi = self(x)[0]
        return np.linalg.lstsq(Phi, y)[0]

    def plot(self, weights, ax=None, num_samples=1000, color='black'):
        """
        Plot the nonlinearity
        """

        x = np.linspace(*(self.tent_span + (num_samples,)))
        y = self(x)[0].dot(weights)

        if ax is None:
            ax = plt.gca()

        ax.plot(x,y,'-',color=color)
        ax.set_xlabel('x', fontsize=22)
        ax.set_ylabel('f(x)', fontsize=22)

        ax.set_xlim(self.tent_span)

        plt.show()
        plt.draw()

        return ax

    @staticmethod
    def append_bias(z, zgrad, zhess=None):
        """
        Add bias term (offset) to tent functions
        """

        # add bias term
        z = np.concatenate((z, np.ones(z.shape[:-1] + (1,))), axis=-1)
        zgrad = np.concatenate((zgrad, np.zeros(z.shape[:-1] + (1,))), axis=-1)

        if zhess is not None:
            zhess = np.concatenate((zhess, np.zeros(zhess.shape[:-1] + (1,))), axis=-1)

        return z, zgrad, zhess


class Gaussian(Nonlinearity):

    def __init__(self, tent_span, num_tents, sigmasq=0.2):
        super(Gaussian, self).__init__(tent_span, num_tents)
        self.sigmasq = sigmasq

    def __call__(self, x, hess=False):

        Phi = np.zeros((x.size, self.centers.size))
        PhiGrad = np.zeros(Phi.shape)

        if hess:
            PhiGrad2 = np.zeros(Phi.shape)

        for idx, c in enumerate(self.centers):
            xc = x.ravel() - c
            z = np.exp(-0.5 * xc ** 2 / self.sigmasq)
            zgrad = - xc / self.sigmasq

            Phi[:, idx] = z
            PhiGrad[:, idx] = z * zgrad
            if hess:
                PhiGrad2[:, idx] = PhiGrad[:, idx] * zgrad - z / self.sigmasq

        z = Phi.reshape(x.shape + (-1,))
        zgrad = PhiGrad.reshape(x.shape + (-1,))
        zhess = PhiGrad2.reshape(x.shape + (-1,)) if hess else None

        return super(Gaussian, self).append_bias(z, zgrad, zhess)


class Linear(Nonlinearity):

    def __init__(self, tent_span, num_tents):
        super(Linear, self).__init__(tent_span, num_tents)

    def __eval__(self, x, hess=False):
        """
        Evaluates gaussian basis functions and derivatives at given values

        Arguments
        ---------
        x        -- the input values at which to evaluate the basis functions
        centers  -- centers used to generate the basis (from _make_linear_basis)

        Returns
        -------
        Phi      -- each of the basis functions evaluated at the locations in x
        PhiGrad  -- derivative of each basis functions evaluated at the locations in x

        """

        Phi = np.zeros((x.size, self.centers.size))
        PhiGrad = np.zeros(Phi.shape)
        xr = x.ravel()

        for idx, c in enumerate(self.centers):

            # left leg of the tent
            if idx > 0:
                il = np.where(((xr >= self.centers[idx - 1]) & (xr <= c)))
                Phi[il, idx] += (xr[il] - self.centers[idx - 1]) / (c - self.centers[idx - 1])
                PhiGrad[il, j] += 1

            # right leg of the tent
            if idx < self.centers.size - 1:
                ir = np.where(( (xr >= self.centers[idx]) & (xr <= self.centers[idx + 1]) ))
                Phi[ir, idx] += (self.centers[idx + 1] - xr[ir]) / (self.centers[idx + 1] - c)
                PhiGrad[ir, idx] -= 1

        z = Phi.reshape(x.shape + (-1,))
        zgrad = PhiGrad.reshape(x.shape + (-1,))
        zhess = np.zeros_like(zgrad) if hess else None

        return super(Linear, self).append_bias(z, zgrad, zhess)


class Ispline(Nonlinearity):

    def __init__(self, tent_span, num_tents, order=3):
        super(Ispline, self).__init__(tent_span, num_tents)
        self.order = order

        # overwrite centers
        centers = np.pad(np.linspace(*(tent_span + (self.num_tents-1,))), self.order-1, 'edge')
        self.centers = np.r_[centers, [centers[-1]]*5]

    def __call__(self, x, hess=False):
        """
        I-spline basis

        TODO: add hessian
        """

        xr = x.ravel()
        Phi = np.vstack([_ispline(xr, self.centers, self.order, idx) for idx in range(self.num_tents)]).T
        PhiGrad = np.vstack([_mspline(xr, self.centers, self.order, idx) for idx in range(self.num_tents)]).T

        z = Phi.reshape(x.shape + (-1,))
        zgrad = PhiGrad.reshape(x.shape + (-1,))
        zhess = None

        return super(Ispline, self).append_bias(z, zgrad, zhess)


def make_rcos_basis(tau, numBases, bias=0.2):
    """
    Makes a raised cosine bases (useful for projecting temporal filters)

    Arguments
    ---------
    tau       -- the time vector along which the bases are generated
    numBases  -- the number of bases to generate. should be smaller than the size of tau
    bias      -- a parameter that can make the time scaling more linear (as bias => Inf) or more skewed (as bias => 0) [default is 0.2]

    Returns
    -------
    Phi       -- the generated basis vectors
    PhiOrth   -- the same basis vectors, but orthogonalized

    """
    from scipy.linalg import orth

    # bias must be nonnegative
    if bias <= 0:
        raise ValueError('Bias term must be positive.')

    # log-scaled time range to place peak centers, plus a factor for stability
    logTime = np.log(tau + bias + 1e-20);

    # centers for basis vectors
    centers = np.linspace(logTime[0], logTime[-1], numBases);

    # make the basis
    Phi = _rcos(logTime.reshape(-1,1), centers.reshape(1,-1), np.mean(np.diff(centers)));

    # return basis and orthogonalized basis
    return Phi, orth(Phi)


def _rcos(x, c, dc):
    """
    The raised cosine function:
    f(x) = 0.5 * cos(u + 1)

    where u is:
    -pi                 if        (x - c)*pi / 2*dc <  -pi
    (x-c)*pi / 2*dc     if -pi <= (x - c)*pi / 2*dc <= -pi
     pi                 if        (x - c)*pi / 2*dc >   pi

    """

    return 0.5*(np.cos(np.maximum(-np.pi,np.minimum(np.pi,0.5*(x-c)*np.pi/dc)))+1)


def _mspline(x, centers, order, idx):
    """
    Generate an mspline function

    Arguments
    ---------
    x           -- x-locations to compute the spline at
    centers     -- x-locations of the knots
    order       -- order of the spline
    idx         -- which spline (knot) to compute the spline for

    Returns
    -------
    y           -- a vector containing the spline values for the corresponding x-locations

    """

    if centers[idx + order] - centers[idx] == 0:
        return 0.0 * x

    if order == 1:
        in_support = (centers[idx] <= x) & (x < centers[idx + 1])
        return in_support * 1 / (centers[idx + 1] - centers[idx] + ~in_support)

    return order * ((x - centers[idx]) * _mspline(x, centers, order - 1, idx) + (centers[idx + order] - x) *
                    _mspline(x, centers, order - 1, idx + 1)) / ((order - 1) * (centers[idx + order] - centers[idx]))


def _ispline(x, centers, order, idx):
    """
    Generate an i-spline function

    Arguments
    ---------
    x           -- x-locations to compute the spline at
    centers     -- x-locations of the knots
    order       -- order of the spline
    idx         -- which spline (knot) to compute the spline for

    Returns
    -------
    y           -- a vector containing the spline values for the corresponding x-locations

    """

    y = np.zeros_like(x)
    for m in range(idx, np.minimum(idx + order, len(centers) - order - 1)):
        y += (centers[m + order + 1] - centers[m]) * _mspline(x, centers, order + 1, m) / (order + 1)
    y[x <= centers[idx]] = 0.0
    y[x > centers[idx + order]] = 1.0
    return y
