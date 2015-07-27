import numpy as np


def cc(r, rhat):
    """
    Correlation coefficient
    """
    return np.corrcoef(np.vstack((rhat, r)))[0, 1]


def lli(r, rhat, meanrate=None):
    """
    log-likelihood improvement over a mean rate model (in bits per spike)
    """
    if meanrate is None:
        meanrate = np.mean(rhat)

    mu = float(np.mean(r * np.log(meanrate) - meanrate))
    return (np.mean(r * np.log(rhat) - rhat) - mu) / (meanrate * np.log(2))


def rmse(r, rhat):
    """
    root mean squared error
    """
    return np.sqrt(np.mean((rhat - r) ** 2))


def fev(r, rhat):
    """
    Fraction of explainable variance
    """
    rate_var = np.mean((np.mean(r) - r) ** 2)

    # fraction of explained variacne
    return 1.0 - (rmse(r, rhat) ** 2) / rate_var
