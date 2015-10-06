"""
Objects for fitting and testing Neural Encoding models

This module provides objects useful for fitting neural encoding models (nems).
Nems are typically probabilistic models of the response of a sensory neuron to
external stimuli. For example, linear-nonlinear-poisson (LNP) or generalized
linear models (GLMs) fall under this category.

The module exposes classes which aid training and testing encoding models given
neural data recorded in an sensory experiment.

Classes
-------
- `NeuralEncodingModel` -- A super class which contains methods that are common
    to all encoding models
- `LNLN` -- A subclass of `NeuralEncodingModel` that fits two layer models
    consisting of alternating layers of linear filtering and nonlinear
    thresholding operations. The parameters for the filter and nonlinearities
    of the first layer are learned, while the linear filter and nonlinearity of
    the second layer are fixed.

References
----------
Coming soon

"""

# standard library imports
import copy
from functools import partial
from collections import defaultdict

# third party packages
import numpy as np
import tableprint
from proxalgs import Optimizer, operators
from sklearn.cross_validation import KFold
from descent import minibatchify


# relative imports
from . import utilities
from . import metrics
from . import nonlinearities
from . import visualization
from . import tentbasis
from .sfo_admm import SFO

# exports
__all__ = ['NeuralEncodingModel', 'LNLN']


class NeuralEncodingModel(object):

    """
    Neural enoding model super class

    Contains helper functions for instantiating, fitting, and testing
    neural encoding models.

    """

    def __init__(self, modeltype, stimulus, spkcounts, filter_dims,
                 minibatch_size, frac_train=0.8, temporal_basis=None):

        # model name / subclass
        self.modeltype = str.lower(modeltype)

        # model properties
        self.theta = None
        self.num_samples = spkcounts.size
        self.tau = filter_dims[-1]
        self.tau_filt = self.tau if temporal_basis is None else temporal_basis.shape[
            1]
        self.filter_dims = filter_dims[:-1] + (self.tau_filt,)
        self.stim_dim = np.prod(self.filter_dims[:-1])

        # the length of the filter must be smaller than the length of the
        # experiment
        assert self.tau <= self.num_samples, 'The temporal filter length must be less than the experiment length.'

        # filter dimensions must be (n1 x n2 x tau), while the stimulus
        # dimensions should be (n1*n2 x t)
        assert stimulus.shape[
            0] == self.stim_dim, 'Stimulus size does not match up with filter dimensions'

        # length of the given firing rate must match the length of the stimulus
        assert stimulus.shape[-
                              1] == spkcounts.size, 'Stimulus length (in time) must equal the length of the given rate'

        # trim the initial portion of the rate (the time shorter than the
        # filter length)
        rate_trim = spkcounts[self.tau:]

        # split data into minibatches
        if minibatch_size is None:

            # choose number of minibatches
            minibatch_size = int(np.ceil(10 * np.sqrt(self.num_samples)))
            num_minibatches = int(self.num_samples / minibatch_size)

        else:
            num_minibatches = int(self.num_samples / minibatch_size)

        # slice the z-scored stimulus every tau samples, for easier dot
        # products
        slices = utilities.rolling_window(
            (stimulus - np.mean(stimulus)) / np.std(stimulus),
            self.tau)

        # store stimulus and rate data for each minibatch in a list
        self.data = list()
        for idx in range(num_minibatches):

            # indices for this minibatch
            minibatch_indices = slice(
                idx * minibatch_size,
                (idx + 1) * minibatch_size)

            # z-score the stimulus and save each minibatch, along with the rate
            # and spikes if given
            if temporal_basis is None:
                self.data.append({
                    'stim': slices[:, minibatch_indices, :],
                    'rate': rate_trim[minibatch_indices]
                })
            else:
                self.data.append({
                    'stim': slices[:, minibatch_indices, :].dot(temporal_basis),
                    'rate': rate_trim[minibatch_indices]
                })

        # set and store random seed (for reproducibility)
        self.random_seed = 12345
        print("Setting random seed to: %i" % self.random_seed)
        np.random.seed(self.random_seed)

        # split up data into train/validation/test sets
        num_train = int(np.round(frac_train * num_minibatches))
        indices = np.arange(num_minibatches)
        self.indices = dict()
        self.indices['train'] = set(
            np.random.choice(
                indices,
                size=num_train,
                replace=False))
        self.indices['test'] = set(indices) - self.indices['train']

        # compute the STA
        self._getsta()

    def __str__(self):
        return "Neural encoding model, " + self.modeltype

    def _getsta(self):
        """
        Compute an STA

        """
        num_samples = float(self.data[0]['rate'].size)
        stas = [
            np.tensordot(
                d['stim'],
                d['rate'],
                axes=(
                    [1],
                    [0])) /
            num_samples for d in self.data]
        self.sta = np.mean(stas, axis=0).reshape(self.filter_dims)

    def add_regularizer(self, theta_key, proxfun, **kwargs):
        """
        Add a proximal operator / objective to the objective, using the proxalgs package

        Parameters
        ----------
        theta_key : string
            The key corresponding to the parameters that this proximal function should be applied to

        proxfun : string
            The name of the corresponding function in the proxalgs.operators module

        \\*\\*kwargs : keyword arguments
            Any keyword arguments required by proxfun

        """

        # ensure regularizers have been initialized
        assert "regularizers" in self.__dict__, "Regularizers dictionary has not been initialized!"
        assert theta_key in self.regularizers, "Key '" + \
            theta_key + "' not found in self.regularizers!"

        # build a wrapper function that applies the desired proximal operator
        # to each element of the parameter array
        def wrapper(theta, rho, **kwargs):

            # creating a copy of the parameters isolates them from any
            # modifications applied by proxfun
            theta_new = copy.deepcopy(theta)

            # apply the proximal operator to each element in the parameter
            # array
            for idx, param in enumerate(theta):
                theta_new[idx] = getattr(
                    operators,
                    proxfun)(
                    param.copy(),
                    float(rho),
                    **kwargs)

            return theta_new

        # add this proximal operator function to the list
        self.regularizers[theta_key].append(partial(wrapper, **kwargs))

    def test(self, theta):
        """
        Evaluate the model on held out data

        Relies on the model subclass having a `metrics` method, which accepts an index into the data array and returns
        a dictionary containing various metrics evaluated given the current value of the parameters (`theta`) on the
        minibatch of data at the given index

        Parameters
        ----------
        theta : dict
            The parameters to test

        Returns
        -------
        train_results : list of tuples
        test_results : list of tuples

        Examples
        --------
        Given an initialized and trained instance of NeuralEncodingModel, you can test the model on
        held out data as follows:

        >>> train_results, test_results = model.test(model.theta)

        """

        # helper function
        def update_results(idx):
            d = self.data[idx]
            rhat = self.rate(theta, d['stim'])[-1]
            return metrics.Score(
                *
                map(lambda k: metrics.__dict__[k](d['rate'], rhat),
                    metrics.Score._fields))

        # evaluate metrics on train / test data
        train = np.mean(
            [update_results(idx)
             for idx in self.indices['train']], axis=0)
        test = np.mean([update_results(idx)
                       for idx in self.indices['test']], axis=0)

        return {'train': train, 'test': test, 'labels': metrics.Score._fields}

    def print_test_results(self, theta):
        """
        Prints a table of test results for the given model parameters

        Parameters
        ----------
        theta : dict
            A dictionary of model parameters

        """

        # run test
        results = self.test(theta)

        # compute averages
        data = [['Test'] + list(results['test']),
                ['Train'] + list(results['train'])]

        # build column headers (names of metrics) and data matrix
        headers = ['Set'] + list(map(str.upper, metrics.Score._fields))

        # print the table
        tableprint.table(data, headers, column_width=10, format_spec='3g')

        return results

    def plot(self):
        visualization.plot(self)

    def kfold(self, nfolds, *args, **kwargs):
        kf = KFold(len(self.data), n_folds=nfolds, shuffle=True)
        results = list()
        for train_indices, test_indices in kf:
            self.indices['train'] = train_indices
            self.indices['test'] = test_indices
            self.fit(*args, **kwargs)
            results.append(self.test(self.theta))
            print('-'*20)
            print('Finished k-fold ({} total)'.format(nfolds))
            print('-'*20)

        return results

    def collect(self, indices):
        """
        Collect firing rates from across minibatches
        """
        from toolz import get

        # sanitize
        if isinstance(indices, str) and indices in ('train', 'test'):
            inds = list(self.indices[indices])
        elif type(indices) in (tuple, list, np.ndarray):
            inds = list(indices)
        else:
            raise ValueError("Input must be an iterable or 'train' or 'test'")

        # compute firing rates
        return np.hstack([(self.rate(self.theta, d['stim'])[-1], d['rate'])
                          for d in get(inds, self.data)])


class LNLN(NeuralEncodingModel):

    def __init__(self, stim, spkcounts, filter_dims, minibatch_size=None,
                 frac_train=0.8, num_subunits=1, num_tents=10, sigmasq=0.5,
                 final_nonlinearity='softrect', num_temporal_bases=None,
                 **kwargs):
        """
        Initializes a two layer cascade linear-nonlinear (LNLN) model

        The model consists of:
        - an initial stage of k different spatiotemporal filters
        - these filtered signals pass through a set of k nonlinearities
        - the output from the k nonlinearities are then summed and put through a final nonlinearity to predict the rate

        The learned parameters include the k first layer spatiotemporal filters and k nonlinearity parameters.
        The nonlinearities are parameterized using a set of tent basis functions

        If k (the number of subunits) is set to 1, the model reduces to a variant of the LN model.

        Examples
        --------

        >>> model = LNLN(stim, rate, filter_dims)

        Parameters
        ----------
        stim : array_like
            a spatiotemporal stimulus. must have dimensions of (N by T), where N is the dimensionality of the
            first linear filter, and T is the number of samples / time points in the experiment

        rate : array_like
            the recorded firing rate corresponding to the stimulus, must have dimensions of (T,)

        filter_dims : tuple
            a tuple defining the dimensions for the spatiotemporal filter. e.g., (n,n,tau) for a 2D stimulus or
            or (n,tau) for a bars stimulus. tau must be less than T, the length of the experiment, and N (the
            stimulus dimensionality) must equal the product of all but the last item in the tuple.

        minibatch_size : int, optional
            the size of each minibatch, in samples. defaults to a value such that the number of minibatches is
            roughly equal to :math:`0.1 * sqrt(T)`

        frac_train : float, optional
            number between 0 and 1, gives the fraction of minibatches used for training (default: 0.8)

        num_subunits : int, optional
            number of subunits to use (default: 1), if a initial W is given, this parameter is unused

        num_tents : int, optional
            number of tent basis functions to use for parameterizing nonlinearities (default: 30)

        sigmasq : float, optional
            the size / scale of each tent basis function (default: 0.2)

        tent_type : string
            the type of tent basis function to use (default: 'gaussian')

        final_nonlinearity : string
            a function from the `nonlinearities` module

        Other Parameters
        ----------------
        spikes : array_like
            an array of spike counts (same dimensions as the rate)

        \\*\\*kwargs : keyword arguments
            if given arguments with the keys `W` or `f`, then those values are used to initialize the filter
            or nonlinearity parameters, respectively.

        """

        # initialize the model object
        if num_temporal_bases is None:
            NeuralEncodingModel.__init__(self, 'lnln', stim, spkcounts,
                                         filter_dims, minibatch_size,
                                         frac_train=frac_train)
        else:
            assert num_temporal_bases < filter_dims[
                -1], "Number of temporal basis functions must be less than the number of temporal dimensions"

            # defaults
            tmax = kwargs['tmax'] if 'tmax' in kwargs else 0.5
            bias = kwargs[
                'temporal_bias'] if 'temporal_bias' in kwargs else 0.2

            # make raised cosine basis
            self.temporal_basis = np.flipud(
                tentbasis.make_rcos_basis
                (np.linspace(0, tmax, filter_dims[-1]), num_temporal_bases,
                 bias=bias)[1])

            # build the reduced model
            NeuralEncodingModel.__init__(self, 'lnln', stim, spkcounts,
                                         filter_dims, minibatch_size,
                                         frac_train=frac_train,
                                         temporal_basis=self.temporal_basis)

        # default # of subunits
        self.num_subunits = kwargs['W'].shape[
            0] if 'W' in kwargs else num_subunits

        # initialize tent basis functions
        tent_span = (-5, 5)          # suitable for z-scored input
        self.tents = tentbasis.Gaussian(tent_span, num_tents, sigmasq=sigmasq)

        # initialize parameter dictionary
        self.theta_init = dict()
        self.theta_init['W'] = np.zeros(
            (self.num_subunits,) + (self.stim_dim, self.tau_filt))
        self.theta_init['f'] = np.zeros(
            (self.num_subunits, self.tents.num_params))

        # initialize filter parameters
        if 'W' in kwargs:

            # check if we need to project onto the temporal basis
            if kwargs['W'].shape[-1] != self.tau_filt:

                if kwargs['W'].shape[-1] == self.tau:
                    kwargs['W'] = kwargs['W'].dot(self.temporal_basis)

                elif kwargs['W'].shape[-1] < self.tau:
                    temp = kwargs['W'].shape[-1]
                    kwargs['W'] = kwargs['W'].dot(
                        self.temporal_basis[
                            :temp,
                            :])

            # ensure dimensions are consistent
            assert self.theta_init['W'].shape == kwargs[
                'W'].shape, "Shape of the filters (`W` keyword argument) " "are inconsistent with the given filter dimensions."

            # normalize each of the given filters
            for idx, w in enumerate(kwargs['W']):
                self.theta_init['W'][idx] = utilities.nrm(w)

        else:

            # multiple subunits: random initialization
            if self.num_subunits > 1:
                for idx in range(self.num_subunits):
                    self.theta_init['W'][idx] = utilities.nrm(
                        0.1 *
                        np.random.randn(
                            self.stim_dim,
                            self.tau_filt))

            # single subunit: initialize with the STA
            else:
                self.theta_init['W'][0] = utilities.nrm(
                    self.sta).reshape(-1, self.sta.shape[-1])

        # initialize nonlinearity parameters
        if 'f' in kwargs:

            # ensure dimensions are consistent
            assert self.theta_init['f'].shape == kwargs['f'].shape, "Shape of the nonlinearity parameters" \
                                                                    " (`f` keyword argument) are inconsistent with " \
                                                                    "the number of tent basis functions."

            self.theta_init['f'] = kwargs['f']

        else:

            # initialize each subunit nonlinearity to be linear
            for idx in range(self.num_subunits):
                ts = self.tents.tent_span
                nonlin_init = np.linspace(ts[0], ts[1], 1000)
                self.theta_init['f'][idx] = self.tents.fit(
                    nonlin_init,
                    nonlin_init)

        # initialize regularizers
        self.regularizers = {'W': list(), 'f': list()}

        # final nonlinearity
        self.final_nonlin_function = getattr(
            nonlinearities,
            final_nonlinearity)

    def f_df(self, W, f, data, param_gradient=None):
        """
        Evaluate the negative log-likelihood objective and gradient for the LNLN model class

        Examples
        --------
        >>> f, df = f_df(self, W, f, data)

        Parameters
        ----------
        W : array_like
            A numpy array containing parameter values for the first layer linear filters in the LNLN model

        f : array_like
            A numpy array containing parameter values for the first layer nonlinearity in the LNLN model

        data : dict
            Dictionary containing two keys: `stim` and `rate`, each of which is a numpy array.

        param_gradient : string (optional, default=None)
            A string indicating which parameters to compute the gradient for, either `W` or `f`

        Returns
        -------
        obj_value : float
            The negative log-likelihood objective value. Lower values indicate a better fit to the data.

        obj_gradient : array_like
            Contains the gradient (as a numpy array) with respect to the parameters given by `param_gradient`

        """

        # f is (K,P)
        # W is (K,N,tau)
        k, n, tau = W.shape
        m = (data['rate'].size - tau + 1)

        # estimate firing rate and get model response
        u, z, zgrad, zhess, drdz, dr2dz2, r = self.rate(
            {'W': W, 'f': f}, data['stim'])

        # objective in bits (poisson log-likelihood)
        obj_value = np.mean(r - data['rate'] * np.log(r))

        # factor in front of the gradient (poisson log-likelihood)
        grad_factor = (1 - data['rate'] / r) * drdz  # dims: (M)

        # compute gradient
        if param_gradient == 'W':
            nonlin_proj = np.sum(
                f[:, np.newaxis, :] * zgrad, axis=2)   # dims: (K, M)
            weighted_proj = grad_factor[
                np.newaxis,
                :] * nonlin_proj  # dims: (K, M)
            obj_gradient = np.tensordot(
                weighted_proj, data['stim'], ([1], [1])) / float(m)

        elif param_gradient == 'f':
            obj_gradient = np.tensordot(grad_factor, z, ([0], [1])) / float(m)

        else:
            obj_gradient = None

        return obj_value, obj_gradient

    def noisy_oracle(self, key, theta_other):

        if key is 'W':
            def f_df_wrapper(theta):
                ind = np.random.choice(list(self.indices['train']), size=1)
                return self.f_df(
                    theta,
                    theta_other,
                    self.data[ind],
                    param_gradient='W')

        elif key is 'f':
            def f_df_wrapper(theta):
                ind = np.random.choice(list(self.indices['train']), size=1)
                return self.f_df(
                    theta_other,
                    theta,
                    self.data[ind],
                    param_gradient='f')

        else:
            raise ValueError('Incorrect key ' + key)

        return f_df_wrapper

    def fit(self,
            num_alt=2,
            max_iter=20,
            num_likelihood_steps=50,
            disp=2,
            check_grad=None,
            callback=None):
        """
        Runs an optimization algorithm to learn the parameters of the model given training data and regularizers

        Parameters
        ----------
        num_alt : int, optional
            The number of times to alternate between optimizing nonlinearities and optimizing filters. Default: 2

        max_iter : int, optional
            The maximum number of steps to take during each leg of the alternating minimization. Default: 25

        num_likelihood_steps : int, optional
            The number of steps to take when optimizing the data likelihood term (using SFO)

        disp : int, optional
            How much information to display during optimization. (Default: 2)

        check_grad : string, optional
            If 'f' or 'W', then the gradient of the log-likelihood objective with respect to that parameter is checked
            against a numerical estimate.

        callback : function
            A callback function that gets called each iteration with the current parameters and a dictionary of other information

        Notes
        -----
        See the `proxalgs` module for more information on the optimization algorithm

        """

        # grab the initial parameters
        theta_current = {
            'W': self.theta_init['W'].copy(),
            'f': self.theta_init['f'].copy()}

        # get list of training data
        train_data = [self.data[idx] for idx in self.indices['train']]

        # data generator
        def datagen():
            while True:
                yield np.random.choice(train_data, 1)[0]

        # store train/test results during optimization
        self.convergence = defaultdict(list)

        def update_results():

            if disp > 0:
                tmp_results = self.print_test_results(theta_current)
                for k in ('train', 'test'):
                    self.convergence[k].append(tmp_results[k])

        # runs the optimization procedure for one set of parameters (a single
        # leg of the alternating minimization)
        def optimize_param(f_df_wrapper, param_key, check_grad, cur_iter):

            # initialize the SFO instance
            loglikelihood_optimizer = SFO(
                f_df_wrapper,
                theta_current[param_key],
                train_data,
                display=0)

            # check gradient
            if check_grad == param_key:
                loglikelihood_optimizer.check_grad()

            # initialize the optimizer object
            opt = Optimizer(
                'sfo',
                optimizer=loglikelihood_optimizer,
                num_steps=num_likelihood_steps)

            # add regularization terms
            [opt.add_regularizer(reg) for reg in self.regularizers[param_key]]

            # run the optimization procedure
            opt.minimize(
                theta_current[param_key],
                max_iter=max_iter,
                disp=disp,
                callback=callback)

            # return parameters and optimization metadata
            return opt.theta

        # print results based on the initial parameters
        print('\n')
        tableprint.table(
            [], ['Initial parameters'], {
                'column_width': 20, 'line_char': '='})

        # print results and store
        update_results()

        # alternating optimization: switch between optimizing nonlinearities,
        # and optimizing filters
        for alt_iter in range(num_alt):

            # Fit filters
            print('\n')
            tableprint.table(
                [], ['Fitting filters'], {
                    'column_width': 20, 'line_char': '='})

            # wrapper for the objective and gradient
            def f_df_wrapper(W, d):
                return self.f_df(W, theta_current['f'], d, param_gradient='W')

            # run the optimization procedure for this parameter
            Wk = optimize_param(
                f_df_wrapper,
                'W',
                check_grad,
                alt_iter +
                0.5).copy()

            # normalize filters
            for filter_index in range(Wk.shape[0]):
                theta_current['W'][filter_index] = utilities.nrm(
                    Wk[filter_index])

            # print and save test results
            update_results()

            # Fit nonlinearity
            print('\n')
            tableprint.table(
                [], ['Fitting nonlinearity'], {
                    'column_width': 20, 'line_char': '='})

            # wrapper for the objective and gradient
            def f_df_wrapper(f, d):
                return self.f_df(theta_current['W'], f, d, param_gradient='f')

            # run the optimization procedure for this parameter
            theta_current['f'] = optimize_param(
                f_df_wrapper,
                'f',
                check_grad,
                alt_iter +
                1).copy()

            # print and save test results
            update_results()

        # store learned parameters
        self.theta = copy.deepcopy(theta_current)

    def hessian(self, theta):
        """
        average the Hessian over all minibatches of data

        """

        L = len(self.data)
        H = self._hessian_minibatch(theta, 0)

        for idx in range(1, L):
            print('{} of {}'.format(idx, L))
            H += self._hessian_minibatch(theta, idx)

        return H / float(L)

    def _hessian_minibatch(self, theta, data_index):
        """
        computes the Hessian of the objective at the given parameter value

        """

        # data
        r = self.data[data_index]['rate']

        # model
        u, z, zgrad, zhess, drdz, dr2dz2, rhat = self.rate(
            theta, self.data[data_index]['stim'], hess=True)

        # store the full Hessian
        N = np.prod(theta['W'].shape[1:])       # filter dimension
        M = theta['W'].shape[0]                 # number of subunits
        P = theta['f'].shape[1]                 # nonlinearity parameters
        T = r.size
        H = np.zeros((N*M + P*M, N*M + P*M))

        # gradient and Hessian factors
        grad_factor = 1 - r/rhat
        hess_factor = r / (rhat**2)

        # nonlinearity projection
        zproj = np.sum(
            theta['f'][
                :,
                np.newaxis,
                :] *
            zgrad,
            axis=2)            # M by T
        zproj2 = np.sum(
            theta['f'][
                :,
                np.newaxis,
                :] * zhess,
            axis=2)           # M by T

        # vectorized data
        x_vec = np.rollaxis(
            self.data[data_index]['stim'], -1).reshape(N, T)      # N x T
        # M*P x T
        z_vec = np.rollaxis(z, -1).reshape(M*P, T)
        scale_factor = np.diag(
            grad_factor *
            dr2dz2 +
            hess_factor *
            drdz**2)    # diagonal(T)

        # nonlinearity - nonlinearity portion
        H[-(M*P):, -(M*P):] = z_vec.dot(scale_factor.dot(z_vec.T))

        # loop over subunits
        for j1 in range(M):

            zgrad_j1 = zproj[j1, :]
            zhess_j1 = zproj2[j1, :]

            # scale factor grad
            sf_zgrad = scale_factor * np.diag(zgrad_j1)

            # subunit-nonlinearity block
            H[j1*N:(j1+1)*N, -(M*P):] = x_vec.dot(sf_zgrad.dot(z_vec.T))
            H[-(M*P):, j1*N:(j1+1)*N] = H[j1*N:(j1+1)*N, -(M*P):].T

            # subunit-subunit block
            for j2 in range(M):

                zgrad_j2 = np.diag(zproj[j2, :])

                # update this portion of the Hessian
                H[j1*N:(j1+1)*N, j2*N:(j2+1)
                  * N] = x_vec.dot((sf_zgrad * zgrad_j2).dot(x_vec.T))

                # diagonal term (for the same subunit)
                if j1 == j2:
                    H[j1 *
                      N:(j1 +
                         1) *
                        N, j2 *
                        N:(j2 +
                           1) *
                        N] += x_vec.dot(np.diag(grad_factor *
                                                drdz *
                                                zhess_j1).dot(x_vec.T))

        return H / float(T)

    def rate(self, theta, stim, hess=False):
        """
        Compute the model response given parameters

        Parameters
        ----------
        theta : dict
        stim : array_like

        Returns
        -------
        u : array_like
        z : array_like
        zgrad : array_like
        zhess : array_like
        drdz : array_like
        r : array_like

        """

        # filter projection
        u = np.tensordot(theta['W'], stim, ([1, 2], [0, 2]))  # dims: (K x M)

        # evaluate input at tent basis functions
        z, zgrad, zhess = self.tents(u, hess=hess)

        # compute log(rate) and the firing rate
        r, drdz, dr2dz2 = self.final_nonlin_function(
            np.tensordot(
                theta['f'], z, ([
                    0, 1], [
                    0, 2])))  # dims: (M)

        return u, z, zgrad, zhess, drdz, dr2dz2, r
