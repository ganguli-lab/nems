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

def build_tents(num_tent_samples, tent_span, num_tents, tent_type='gaussian', sigmasq=0.2):
    """
    Initialize tent basis functions
    (used to parameterize nonlinearities)

    Arguments
    ---------
    num_tent_samples    -- number of samples for the input range (tent_x)
    tent_span           -- tuple designating the input range e.g. (min, max)
    num_tents           -- number of tent basis functions to use
    tent_type           -- shape of the basis functions (defaults to 'gaussian'. other option: 'linear')

    Returns
    -------
    tentparams          -- a dictionary containing a bunch of tent basis function parameters:
                            num_tent_samples, tent_span, num_tents, tent_type, sigmasq (only used for Gaussian tents),
                            Phi (a matrix whose columns are the basis functions), and centers (an array of tent centers)

    """

    # store tent basis function parameters in a dictionary
    tentparams = dict()
    tentparams['num_tent_samples'] = num_tent_samples
    tentparams['tent_span'] = tent_span
    tentparams['num_tents'] = num_tents + 1
    tentparams['type'] = tent_type
    tentparams['sigmasq'] = sigmasq

    # initialize x-axis for basis functions
    tentparams['tent_x'] = np.linspace(tent_span[0], tent_span[1], num_tent_samples)

    # build tent basis functions
    if str.lower(tent_type) == 'gaussian':
        tentparams['Phi'], tentparams['centers'] = _make_gaussian_basis(tentparams['tent_x'], num_tents, sigmasq=sigmasq)

    elif str.lower(tent_type) == 'linear':
        tentparams['Phi'], tentparams['centers'] = _make_linear_basis(tentparams['tent_x'], num_tents)

    elif str.lower(tent_type) == 'ispline':
        tentparams['Phi'], tentparams['centers'] = _make_ispline_basis(tentparams['tent_x'], num_tents)

    else:
        raise ValueError('Could not parse tent type ' + tent_type)

    # return tent function parameters
    return tentparams

def eval_tents(u, tentparams, hess=False):
    """
    Evaluate basis functions and derivative at given input value

    Arguments
    ---------
    u           -- input to evaluate
    tentparams  -- parameters used to specify the tent basis functions (e.g. returned by build_tents)

    Returns
    -------
    z, zgrad    --

    """

    # evaluate the tent basis functions
    if str.lower(tentparams['type']) == 'gaussian':

        if hess:
            z, zgrad, zhess = _eval_gaussian_basis(u, tentparams['centers'], tentparams['sigmasq'], hess=True)

        else:
            z, zgrad = _eval_gaussian_basis(u, tentparams['centers'], tentparams['sigmasq'])

    elif str.lower(tentparams['type']) == 'linear':
        z, zgrad = _eval_linear_basis(u, tentparams['centers'])

    elif str.lower(tentparams['type']) == 'ispline':
        z, zgrad = _eval_ispline_basis(u, tentparams['centers'], tentparams['num_tents']-1)

    else:
        raise ValueError('Could not parse tent type ' + tentparams['type'])

    # add bias term
    z = np.concatenate((z, np.ones(z.shape[:-1] + (1,))), axis=-1)
    zgrad = np.concatenate((zgrad, np.zeros(z.shape[:-1] + (1,))), axis=-1)

    if hess:
        zhess = np.concatenate((zhess, np.zeros(zhess.shape[:-1] + (1,))), axis=-1)
        return z, zgrad, zhess

    else:
        return z, zgrad, None

def _make_ispline_basis(x, numBases, order=3, limits=None):
    """
    Generate a basis of piecewise linear functions

    Arguments
    ---------
    x           -- a vector spanning an input range. e.g. x = np.linspace(-5,5,1e3)
    numBases    -- the number of tent basis functions to generate
    order       -- order of the mspline
    limits      -- an optional tuple which sets the range of the tent functions. Defaults to (x[0], x[-1])

    Returns
    -------
    Phi         -- a matrix consisting of columns that are the I-splines over the input range
    centers     -- parameters needed to specify this exact set of tent functions (used by _eval_ispline_basis)

    """

    if limits is None:
        limits = np.array([x[0], x[-1]])

    # knot centers
    centers = np.pad(np.linspace(limits[0], limits[1], numBases-1), order-1, 'edge')
    centers = np.r_[centers, [centers[-1]]*5]

    # build basis functions
    Phi = np.vstack([_ispline(x, centers, order, idx) for idx in range(numBases)]).T

    return Phi, centers

def _eval_ispline_basis(x, centers, numBases, order=3):
    """
    Evaluates i-spline basis functions and derivatives at given values

    Arguments
    ---------
    x        -- the input values at which to evaluate the basis functions
    centers  -- centers used to generate the basis (from _make_ispline_basis)

    Returns
    -------
    Phi      -- each of the basis functions evaluated at the locations in x
    PhiGrad  -- derivative of each basis functions evaluated at the locations in x

    """

    xr = x.ravel()
    Phi = np.vstack([_ispline(xr, centers, order, idx) for idx in range(numBases)]).T
    PhiGrad = np.vstack([_mspline(xr, centers, order, idx) for idx in range(numBases)]).T

    return Phi.reshape(x.shape + (-1,)), PhiGrad.reshape(x.shape + (-1,))

def _make_linear_basis(x, numBases, limits=None):
    """Generate a basis of piecewise linear functions

    Arguments
    ---------
    x           -- a vector spanning an input range. e.g. x = np.linspace(-5,5,1e3)
    numBases    -- the number of tent basis functions to generate
    limits      -- an optional tuple which sets the range of the tent functions. Defaults to (x[0], x[-1])

    Returns
    -------
    Phi         -- a matrix consisting of columns that are the tent functions over the input range
    centers     -- parameters needed to specify this exact set of tent functions (used by _eval_linear_basis)

    """

    if limits is None:
        limits = np.array([x[0], x[-1]])

    # centers of the basis functions
    centers = np.linspace(limits[0], limits[1], numBases)

    # build basis functions
    dim = x.size
    Phi = np.zeros((dim, numBases))
    for j in range(0,numBases):

        # left leg of the tent
        if j > 0:
            il = np.where(( (x >= centers[j-1]) & (x <= centers[j]) ))
            Phi[il,j] += (x[il]-centers[j-1]) / (centers[j] - centers[j-1])

        # right leg of the tent
        if j < numBases-1:
            ir = np.where(( (x >= centers[j]) & (x <= centers[j + 1]) ))
            Phi[ir,j] += (centers[j+1]-x[ir]) / (centers[j+1] - centers[j])

    return Phi, centers

def _eval_linear_basis(x, centers):
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

    Phi = np.zeros((x.size, centers.size))
    PhiGrad = np.zeros(Phi.shape)
    xr = x.ravel()

    for j in range(centers.size):

        # left leg of the tent
        if j > 0:
            il = np.where(( (xr >= centers[j - 1]) & (xr <= centers[j]) ))
            Phi[il, j] += (xr[il] - centers[j - 1]) / (centers[j] - centers[j - 1])
            PhiGrad[il, j] += 1

        # right leg of the tent
        if j < centers.size - 1:
            ir = np.where(( (xr >= centers[j]) & (xr <= centers[j + 1]) ))
            Phi[ir, j] += (centers[j + 1] - xr[ir]) / (centers[j + 1] - centers[j])
            PhiGrad[ir, j] += -1

    return Phi.reshape(x.shape + (-1,)), PhiGrad.reshape(x.shape + (-1,))  # , PhiGrad2

def _make_gaussian_basis(x, numBases, sigmasq=0.2, limits=None):
    """
    Generates a basis of Gaussian tent functions

    Arguments
    ---------
    x           -- a vectors spanning an input range. e.g. x = np.linspace(-5,5,1e3)
    numBases    -- the number of tent basis functions to generate
    sigmasq     -- std. dev. of the gaussian tents (default is 0.2)
    limits      -- an optional tuple which sets the range of the tent functions. Defaults to (x[0], x[-1])

    Returns
    -------
    Phi         -- the generated basis vectors (each basis vector is a column of Phi)
    params      -- tuple of parameters (centers, sigmasq) used to generate the basis (useful for _eval_gaussian_basis)

    """

    if limits is None:
        limits = np.array([x[0], x[-1]])

    # centers of the basis functions
    centers = np.linspace(limits[0], limits[1], numBases)

    # build basis functions
    dim = x.size
    Phi = np.zeros((dim,numBases))
    for j in range(numBases):
        Phi[:,j] = np.exp( -0.5*(x - centers[j])**2 / sigmasq )

    return Phi, centers

def _eval_gaussian_basis(x, centers, sigmasq, hess=False):
    """
    Evaluates gaussian basis functions and derivatives at given values

    Arguments
    ---------
    x        -- the input values at which to evaluate the basis functions
    centers  -- array of centers used to generate the basis (from _make_gaussian_basis)
    sigmasq  -- the width (std. dev. of each basis function)

    Returns
    -------
    Phi      -- each of the basis functions evaluated at the locations in x
    PhiGrad  -- derivative of each basis functions evaluated at the locations in x

    """

    Phi = np.zeros((x.size, centers.size))
    PhiGrad = np.zeros(Phi.shape)
    if hess:
        PhiGrad2 = np.zeros(Phi.shape)        # uncomment the PhiGrad2 lines if you need the second derivative

    for j in range(centers.size):
        z = np.exp( -0.5 * (x.ravel() - centers[j])**2 / sigmasq )
        zgrad = - (x.ravel() - centers[j]) / sigmasq

        Phi[:,j] = z
        PhiGrad[:,j] = z * zgrad
        if hess:
            PhiGrad2[:,j] = PhiGrad[:,j] * zgrad - z / sigmasq

    if hess:
        return Phi.reshape(x.shape + (-1,)), PhiGrad.reshape(x.shape + (-1,)), PhiGrad2.reshape(x.shape + (-1,))

    else:
        return Phi.reshape(x.shape + (-1,)), PhiGrad.reshape(x.shape + (-1,))

def eval_basis(u, x, Phi, PhiGrad):
    """
    Evaluates a function composed of basis functions given by vectors Phi,
    and derivatives given by vectors PhiGrad.

    Arguments
    ---------
    u       -- the input to evaluate
    x       -- the input range of the basis vectors
    Phi     -- the basis vectors (columns of Phi)
    PhiGrad -- the derivative of the basis vectors (columns of PhiGrad)

    Returns
    -------
    z       -- the input (u) evaluated at each basis function
    zgrad   -- the input (u) evaluated at each basis function derivative

    """

    inds = _find_nearest(x, u)
    return Phi[inds, :], PhiGrad[inds, :]

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

def _find_nearest(x, values):
    """
    Finds the nearest index of given values to an array location

    Arguments
    ---------
    x       -- the discretized input range we wish to approximate
    values  -- the (arbitrary precision) input values

    Returns
    -------
    indices -- an array (same size as values) containing the nearest index of each value to x

    """
    return np.array([np.abs(x-v).argmin() for v in values.ravel()]).reshape(values.shape)

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