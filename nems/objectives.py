"""
Objective functions
"""
import numpy as np


def poisson_loglikelihood(theta, data, theta_fixed, modelrate, dt, regularizers):
    """
    requirements:

    data['rate'] is (1,m) for m time points
    data['stim'] is (dim, m)

    modelrate is a function that returns: the estimated rate r for k cells at m
      time points with dimensions: (k, m) and gradient, dr, evaluated at theta.
      each element of dr must have dimensions: (d1, d2, k, m) or (d, k, m)

    dt is a time step (scalar)

    regularizers is a dictionary with the same keys as theta
    regularizers[key] is a function that takes as an argument the parameter value
      theta[key] and returns the regularization penalty (scalar) and gradient
      (same size as theta[key]) at that particular parameter value
    """
    # compute model firing rate
    theta_fixed.update(theta)
    logr, r, dr = modelrate(theta_fixed, data, keys=theta.keys())

    # poisson log-likelihood
    eps = 1e-12
    f = np.mean(r * dt - data['rate'] * logr)
    fgrad = (dt - data['rate'] / (r + eps))                     # (k, m)
    T = float(data['rate'].size)

    # gradient
    df = dict()
    for key in theta:

        # add regularization for this parameter
        penalty, pgrad = regularizers[key](theta[key])
        f += penalty

        # ganglion cell filter (depends on the cell index, k)
        if dr[key].ndim == 3:
            df[key] = np.squeeze(np.sum(dr[key] * fgrad.reshape(1, fgrad.shape[0], -1), axis=2)) / T + pgrad

        # other parameters (sum over the number of cells, k)
        elif dr[key].ndim == 4:
            df[key] = np.tensordot(dr[key], fgrad, ([2, 3], [0, 1])) / T + pgrad  # dims: dr[key].shape[:2]

        else:
            raise ValueError('The number of dimensions of each value in the gradient (dr) needs to be 3 or 4')

    return f, df
