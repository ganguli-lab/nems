"""
Objects for fitting and testing Neural Encoding models

This module provides a set of objects useful for fitting neural encoding models. These are typically probabilistic
models of the response of a sensory neuron to an external stimulus. For example, linear-nonlinear-poisson (LNP) or
generalized linear models (GLMs) fall under this category.

The module exposes classes which contain methods useful for training and testing encoding models given neural data
recorded in an experiment. For more information, see the documentation (TODO: website)

Classes
-------
- `NeuralEncodingModel` -- A super class which contains methods that are common to all encoding models
- `LNLN` -- A subclass of `NeuralEncodingModel` that fits two layer models consisting of alternating layers of linear filtering and nonlinear thresholding operations. The parameters for the filter and nonlinearities of the first layer are learned, while the linear filter and nonlinearity of the second layer are fixed.
- `LN` -- A subclass of `NeuralEncodingModel` that fits linear-nonlinear (LN) models using a Gaussian log-likelihood.

References
----------
Coming soon

"""

# imports
import time
import copy
from functools import partial
import numpy as np
import tentbasis
import nonlinearities
import utilities
from sfo_admm import SFO
from proxalgs.core import Optimizer
from proxalgs import operators
import pandas as pd
import tableprint

# exports
__all__ = ['NeuralEncodingModel', 'LNLN']


class NeuralEncodingModel(object):
    """
    Neural enoding model object

    """

    def __init__(self, modeltype, stimulus, rate, filter_dims, minibatch_size, frac_train=0.8, spikes=None):

        # model name / subclass
        self.modeltype = str.lower(modeltype)

        # model properties
        self.theta = None
        self.num_samples = rate.size
        self.tau = filter_dims[-1]
        self.filter_dims = filter_dims
        self.stim_dim = np.prod(filter_dims[:-1])

        # the length of the filter must be smaller than the length of the experiment
        assert self.tau <= self.num_samples, 'The temporal filter length must be less than the experiment length.'

        # filter dimensions must be (n1 x n2 x tau), while the stimulus dimensions should be (n1*n2 x t)
        assert stimulus.shape[0] == self.stim_dim, 'Stimulus size does not match up with filter dimensions'

        # split data into minibatches
        if minibatch_size is None:

            # choose number of minibatches
            minibatch_size = np.ceil(10 * np.sqrt(self.num_samples)).astype('int')
            num_minibatches = int(self.num_samples / minibatch_size)

        else:
            num_minibatches = int(self.num_samples / minibatch_size)

        # slice the z-scored stimulus every tau samples, for easier dot products
        slices = utilities.rolling_window((stimulus - np.mean(stimulus)) / np.std(stimulus), self.tau)

        # store stimulus and rate data for each minibatch in a list
        self.data = list()
        for idx in range(num_minibatches):

            # indices for this minibatch
            minibatch_indices = slice(idx * minibatch_size, (idx + 1) * minibatch_size)

            # z-score the stimulus and save each minibatch, along with the rate and spikes if given
            if spikes is not None:
                self.data.append({
                    'stim': slices[:, minibatch_indices, :],
                    'rate': rate[minibatch_indices],
                    'spikes': np.where(spikes[minibatch_indices] > 0)[0]
                })

            else:
                self.data.append({
                    'stim': slices[:, minibatch_indices, :],
                    'rate': rate[minibatch_indices]
                })

        # set and store random seed (for reproducibility)
        self.random_seed = np.random.randint(1e5)
        np.random.seed(self.random_seed)

        # split up data into train/validation/test sets
        num_train = int(np.round(frac_train * num_minibatches))
        indices = np.arange(num_minibatches)
        np.random.shuffle(indices)
        self.train_indices = indices[:num_train]
        self.test_indices = indices[num_train:]

        # compute the STA
        self._getsta()

        # compute firing rate statistics
        rate_concat = np.hstack([d['rate'] for d in self.data])
        self.meanrate = np.mean(rate_concat)
        self.rate_variance = np.var(rate_concat)

        # store metadata and convergence information in pandas DataFrames
        self.metadata = pd.DataFrame()
        self.convergence = pd.DataFrame()

    def __str__(self):
        return "Neural encoding model, " + self.modeltype

    def _getsta(self):
        """
        Compute an STA

        """
        num_samples = float(self.data[0]['rate'].size)
        stas = [np.tensordot(d['stim'], d['rate'], axes=([1], [0])) / num_samples for d in self.data]
        self.sta = np.mean(stas,axis=0).reshape(self.filter_dims)

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
        assert theta_key in self.regularizers, "Key '" + theta_key + "' not found in self.regularizers!"

        # build a wrapper function that applies the desired proximal operator to each element of the parameter array
        def wrapper(theta, rho, **kwargs):

            # creating a copy of the parameters isolates them from any modifications applied by proxfun
            theta_new = copy.deepcopy(theta)

            # apply the proximal operator to each element in the parameter array
            for idx, param in enumerate(theta):
                theta_new[idx] = getattr(operators, proxfun)(param.copy(), float(rho), **kwargs)

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
        results : DataFrame
            A pandas dataframe with the following columsn: the minibatch index, a 'train' or 'test' label, and any
            keywords returned by the metrics function. Each row in the dataframe consists of the metrics evaluated
            on a single minibatch of data.

        stats : dict
            Statistics on just the held out minibatches. Keys correspond to the keys in metrics, values are the average
            computed over the test minibatches.

        Examples
        --------
        Given an initialized and trained instance of NeuralEncodingModel, you can test the model on
        held out data as follows:

        >>> results, avg = model.test(model.theta)

        """

        results = list()
        indices = list()
        class_key = 'set'

        # helper function
        def update_results(idx, class_label):
            res = self.metrics(idx, theta)
            res.update({class_key: class_label})
            results.append(res)
            indices.append(idx)

        # evaluate metrics on train / test data
        [update_results(idx, 'train') for idx in self.train_indices]
        [update_results(idx, 'test') for idx in self.test_indices]

        # create DataFrame to store results
        df = pd.DataFrame(data=results, index=indices)

        # compute the average over the test minibatches
        avg = dict(df.loc[df[class_key] == 'test'].mean())

        return df, avg

    def print_test_results(self, theta):
        """
        Prints a table of test results for the given model parameters

        Parameters
        ----------
        theta : dict
            A dictionary of model parameters

        """

        # run test
        df, test_res = self.test(theta)

        # also compute training results, and spreads
        ntrain = (df['set'] == 'train').sum()
        ntest = (df['set'] == 'test').sum()
        train_res = dict(df.loc[df['set'] == 'train'].mean())
        train_spread = dict(df.loc[df['set'] == 'train'].std())
        test_spread = dict(df.loc[df['set'] == 'test'].std())

        # build column headers (names of metrics) and data matrix
        headers = ['Train/Test Set']
        data = [['train'], ['test']]

        for key in test_res.keys():

            # headers
            headers += [key + ' (mean)']
            headers += [key + ' (sem)']

            # data (mean + standard error of the mean)
            data[0] += [train_res[key], train_spread[key] / np.sqrt(ntrain)]
            data[1] += [test_res[key], test_spread[key] / np.sqrt(ntest)]

        # print the table
        tableprint.table(data, headers, {'column_width': 30, 'precision': '8g'})

        return df


class LNLN(NeuralEncodingModel):
    def __init__(self, stim, rate, filter_dims, minibatch_size=None, frac_train=0.8, num_subunits=1,
                 num_tents=30, sigmasq=0.2, tent_type='gaussian', final_nonlinearity='softrect', spikes=None, **kwargs):
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
            number of tent basis functions to use for parameterizing nonlinearities

        sigmasq : float, optional
            the size / scale of each tent basis function (default: 0.2)

        tent_type : string
            the type of tent basis function to use (default: 'Gaussian')

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
        NeuralEncodingModel.__init__(self, 'lnln_exp', stim, rate, filter_dims, minibatch_size,
                                     frac_train=frac_train, spikes=spikes)

        # default # of subunits
        if 'W' in kwargs:
            self.num_subunits = kwargs['W'].shape[0]
        else:
            self.num_subunits = num_subunits

        # initialize tent basis functions
        num_tent_samples = 1000
        tent_span = (-5,5)          # suitable for z-scored input
        self.tentparams = tentbasis.build_tents(num_tent_samples, tent_span, num_tents,
                                                tent_type=tent_type, sigmasq=sigmasq)

        # initialize parameter dictionary
        self.theta_init = dict()
        self.theta_init['W'] = np.zeros((self.num_subunits,) + (self.stim_dim, self.tau))
        self.theta_init['f'] = np.zeros((self.num_subunits, self.tentparams['num_tents']))

        # initialize filter parameters
        if 'W' in kwargs:

            # ensure dimensions are consistent
            assert self.theta_init['W'].shape == kwargs['W'].shape, "Shape of the filters (`W` keyword argument) " \
                                                                    "are inconsistent with the given filter dimensions."

            # normalize each of the given filters
            for idx, w in enumerate(kwargs['W']):
                self.theta_init['W'][idx] = utilities.nrm(w)

        else:

            # multiple subunits: random initialization
            if self.num_subunits > 1:
                for idx in range(self.num_subunits):
                    self.theta_init['W'][idx] = utilities.nrm(0.1 * np.random.randn(self.stim_dim, self.tau))

            # single subunit: initialize with the STA
            else:
                self.theta_init['W'][0] = utilities.nrm(self.sta).reshape(-1,self.sta.shape[-1])

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
                ts = self.tentparams['tent_span']
                nonlin_init = np.linspace(ts[0], ts[1], self.tentparams['num_tent_samples'])
                self.theta_init['f'][idx,:] = np.append(np.linalg.lstsq(self.tentparams['Phi'], nonlin_init)[0], 0)

        # initialize regularizers
        self.regularizers = {'W': list(), 'f': list()}

        # final nonlinearity
        self.final_nonlin_function = getattr(nonlinearities, final_nonlinearity)

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
        u, z, zgrad, drdz, r = self.rate({'W': W, 'f': f}, data['stim'])

        # objective in bits (poisson log-likelihood)
        obj_value = np.mean(r - data['rate'] * np.log(r))

        # factor in front of the gradient (poisson log-likelihood)
        grad_factor = (1 - data['rate'] / r) * drdz  # dims: (M)

        # compute gradient
        if param_gradient == 'W':
            nonlin_proj = np.sum(f[:, np.newaxis, :] * zgrad, axis=2)  # dims: (K, M)
            weighted_proj = grad_factor[np.newaxis, :] * nonlin_proj  # dims: (K, M)
            obj_gradient = np.tensordot(weighted_proj, data['stim'], ([1], [1])) / float(m)

        elif param_gradient == 'f':
            obj_gradient = np.tensordot(grad_factor, z, ([0], [1])) / float(m)

        else:
            obj_gradient = None

        return obj_value, obj_gradient

    def fit(self, num_alt=2, max_iter=20, num_likelihood_steps=50, disp=2, check_grad=None):
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

        Notes
        -----
        See the `proxalgs` module for more information on the optimization algorithm

        """

        # reset the metadata and convergence data structures
        self.metadata = pd.DataFrame()
        self.convergence = pd.DataFrame()

        # grab the initial parameters
        theta_current = {'W': self.theta_init['W'].copy(), 'f': self.theta_init['f'].copy()}

        # get list of training data
        train_data = [self.data[idx] for idx in self.train_indices]

        # runs the optimization procedure for one set of parameters (a single leg of the alternating minimization)
        def optimize_param(f_df_wrapper, param_key, check_grad, cur_iter):

            # initialize the SFO instance
            loglikelihood_optimizer = SFO(f_df_wrapper, theta_current[param_key], train_data, display=0)

            # check gradient
            if check_grad == param_key:
                loglikelihood_optimizer.check_grad()

            # initialize the optimizer object
            opt = Optimizer('sfo', optimizer=loglikelihood_optimizer, num_steps=num_likelihood_steps)

            # add regularization terms
            [opt.add_regularizer(reg) for reg in self.regularizers[param_key]]

            # run the optimization procedure
            opt.minimize(theta_current[param_key], max_iter=max_iter, disp=disp)

            # add metadata to converge dataframe
            opt.metadata['Iteration'] = cur_iter
            self.metadata = self.metadata.append(opt.metadata)

            # return parameters and optimization metadata
            return opt.theta

        # print results based on the initial parameters
        print('\n')
        tableprint.table([], ['Initial parameters'], {'column_width': 20, 'line_char': '='})
        self.print_test_results(theta_current)

        # alternating optimization: switch between optimizing nonlinearities, and optimizing filters
        for alt_iter in range(num_alt):

            # Fit nonlinearity
            print('\n')
            tableprint.table([], ['Fitting nonlinearity'], {'column_width': 20, 'line_char': '='})

            # wrapper for the objective and gradient
            def f_df_wrapper(f, d):
                return self.f_df(theta_current['W'], f, d, param_gradient='f')

            # run the optimization procedure for this parameter
            theta_current['f'] = optimize_param(f_df_wrapper, 'f', check_grad, alt_iter + 0.5).copy()

            # run the callback
            df = self.print_test_results(theta_current)
            avg = df.groupby('set').mean()
            avg['Iteration'] = alt_iter + 0.5
            self.convergence = self.convergence.append(avg)

            # Fit filters
            print('\n')
            tableprint.table([], ['Fitting filters'], {'column_width': 20, 'line_char': '='})

            # wrapper for the objective and gradient
            def f_df_wrapper(W, d):
                return self.f_df(W, theta_current['f'], d, param_gradient='W')

            # run the optimization procedure for this parameter
            Wk = optimize_param(f_df_wrapper, 'W', check_grad, alt_iter + 1).copy()

            # normalize filters
            for filter_index in range(Wk.shape[0]):
                theta_current['W'][filter_index] = utilities.nrm(Wk[filter_index])

            # print and save test results
            df = self.print_test_results(theta_current)
            avg = df.groupby('set').mean()
            avg['Iteration'] = alt_iter + 1
            self.convergence = self.convergence.append(avg)

        # store learned parameters
        self.theta = copy.deepcopy(theta_current)

    def metrics(self, data_index, theta):
        """
        Evaluate metrics on a given minibatch.

        Parameters
        ----------
        data_index : int
            An index into the array of minibatches (`self.data`) the evaluate

        theta : dict
            The parameters to use in the evaluation

        Returns
        -------
        metrics : dict
            A dictionary whose keys are the names of metrics applied to evaluate the model parameters on the given
            minibatch, and whose values are single numbers.

        See Also
        --------
        NeuralEncodingModel.test
            The `test` function in the super class NeuralEncodingModel relies on the metrics function to return
            aggregate statistics over all minibatches.

        """

        # get this minibatch of data
        data = self.data[data_index]

        # compute the model firing rate
        rhat = self.rate(theta, data['stim'])[-1]
        logr = np.log(rhat)

        # correlation coefficient
        cc = float(np.corrcoef(np.vstack((rhat, data['rate'])))[0, 1])

        # log-likelihood improvement over a mean rate model (in bits per spike)
        mu = float(np.mean(data['rate'] * np.log(self.meanrate) - self.meanrate))
        fobj = (float(np.mean(data['rate'] * logr - rhat)) - mu) / (self.meanrate * np.log(2))

        # mean squared error
        mse = float(np.mean((rhat - data['rate']) ** 2))

        # fraction of explained variacne
        fev = 1.0 - (mse / self.rate_variance)

        return {'corrcoef': cc, 'log-likelihood (rel.)': fobj,
                'mean squared error': mse, 'fraction of explained variance': fev}

    def _stim_gradient(self, stim):
        """
        Compute the model response and gradient with respect to the stimulus

        .. warning:: Work in progress

        """

        u, z, zgrad, drdz, r = self.rate(self.theta, stim)

        xgrad = np.zeros_like(np.squeeze(stim))
        for idx in range(self.theta['W'].shape[0]):

            xgrad += self.theta['W'][idx] * zgrad[idx, 0, :].dot(self.theta['f'][idx, :])

        # fgrad = np.tensordot(zgrad, self.theta['f'], ([0,2],[0,1]))
        # wgrad = np.tensordot(u, self.theta['W'],([0],[0]))

        return r, xgrad * r

    def rate(self, theta, stim):
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
        drdz : array_like
        r : array_like

        """

        # filter projection
        u = np.tensordot(theta['W'], stim, ([1, 2], [0, 2]))  # dims: (K x M)

        # evaluate input at tent basis functions
        z, zgrad = tentbasis.eval_tents(u, self.tentparams)

        # compute log(rate) and the firing rate
        r, drdz = self.final_nonlin_function(np.tensordot(theta['f'], z, ([0, 1], [0, 2])))  # dims: (M)

        return u, z, zgrad, drdz, r
