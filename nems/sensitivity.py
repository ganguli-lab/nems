"""
functions to help with sensitivity analysis

.. warning:: Work in progress

"""

__author__ = 'nirum'

def lnln_hessian(self, theta, data):
    """
    hessian of the LNLN model
    """
    dim, m = data['stim'].shape
    p = theta['f'].shape[0]

    # copy and zscore
    stim = zscore(data['stim'].astype('float64'))
    m, dim = stim.shape

    u = stim.dot(theta['W'])
    z, zgrad, zgrad2 = tentbasis.evalGaussianBasis(u, self.tentparams)  # m by k by p
    logr = (z.reshape(m, -1).dot(theta['f'].T.reshape(-1, 1))).reshape(1, -1)  # 1 by m
    r = np.exp(logr)  # 1 by m

    # initialize
    num_subunits = z.shape[1]
    Hess = dict()
    Hess['W'] = np.zeros((dim * num_subunits, dim * num_subunits))

    # nonlinearities
    Hess['f'] = (z.reshape(m, -1).T * r).dot(z.reshape(m, -1))

    # loop over subunits
    for rowidx in range(num_subunits):
        row_inds = slice(rowidx * dim, (rowidx + 1) * dim)
        for colidx in range(num_subunits):
            col_inds = slice(colidx * dim, (colidx + 1) * dim)

            # subunits
            zg_a = np.squeeze(zgrad[:, rowidx, :].dot(theta['f'][:, rowidx]))  # m by 1
            zg_b = np.squeeze(zgrad[:, colidx, :].dot(theta['f'][:, colidx]))

            if rowidx == colidx:
                zg2 = np.squeeze(zgrad2[:, rowidx, :].dot(theta['f'][:, rowidx]))
                weights = zg_a * zg_b * np.squeeze(r)  # + zg2 * np.squeeze(r - data['rate'])
            else:
                weights = zg_a * zg_b * np.squeeze(r)

            Hess['W'][row_inds, col_inds] = data['stim'].dot((data['stim'].T * weights.reshape(-1, 1)))

    return Hess

def ln_hessian(self, theta, data):
    """
    hessian of the LNLN model
    """

    # copy and zscore
    stim = zscore(data['stim'].astype('float64'))

    m, dim = stim.shape

    u = stim.dot(theta['W'])
    z, zgrad, zgrad2 = tentbasis.evalGaussianBasis(u, self.tentparams)  # m by 1 by p
    logr = (z.reshape(m, -1).dot(theta['f'].T.reshape(-1, 1))).reshape(1, -1)  # 1 by m
    r = np.exp(logr)  # 1 by m

    # compute Hessian
    Hess = dict()

    # filter
    zg = np.squeeze(zgrad.dot(theta['f']))
    zg2 = np.squeeze(zgrad2.dot(theta['f']))
    weights = (r * (zg - zg2) * (self.dt - np.squeeze(data['rate']) / r) + (np.squeeze(data['rate']) / (r ** 2)) * (
        r * zg) ** 2)
    Hess['W'] = data['stim'].dot((data['stim'].T * weights.reshape(-1, 1)))

    # nonlinearity
    Hess['f'] = (np.squeeze(z).T * r).dot(np.squeeze(z))

    return Hess
