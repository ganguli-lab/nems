"""
Nonlinearities and their derivatives

Each function returns the value and derivative of a nonlinearity. Given :math:`y = f(x)`, the function returns
:math:`y` and :math:`dy/dx`

"""

import numpy as np


def exp(x):
    """
    An exponential function

    Parameters
    ----------
    x : array_like
        Input values

    Returns
    -------
    y : array_like
        The nonlinearity evaluated at the input

    dydx : array_like
        The derivative of the nonlinearity evaluated at the input

    """

    # compute the exponential
    y = np.exp(x)

    # compute the first and second derivatives
    dydx = y
    dy2dx2 = y

    return y, dydx, dy2dx2


def softrect(x):
    """
    Soft rectifying function

    .. math::
        y = \log(1+e^x)

    Parameters
    ----------
    x : array_like
        Input values

    Returns
    -------
    y : array_like
        The nonlinearity evaluated at the input

    dydx : array_like
        The derivative of the nonlinearity evaluated at the input

    """

    # compute the soft rectifying nonlinearity
    x_exp = np.exp(x)
    y = np.log(1 + x_exp)

    # compute the derivative
    dydx = x_exp / (1 + x_exp)

    # compute the second derivative
    dy2dx2 = x_exp / (1 + x_exp)**2

    return y, dydx, dy2dx2


def linear(x):
    """
    A linear function

    Parameters
    ----------
    x : array_like
        Input values

    Returns
    -------
    y : array_like
        The nonlinearity evaluated at the input

    dydx : array_like
        The derivative of the nonlinearity evaluated at the input

    """

    # linear function leaves variable unchanged
    y = x

    # compute the derivative (constant) and second derivative (zero)
    dydx = np.ones(x.shape)
    dy2dx2 = np.zeros(x.shape)

    return y, dydx, dy2dx2
