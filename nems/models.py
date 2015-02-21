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

References
----------
Coming soon

"""

# imports
import time
import copy
from functools import partial
import numpy as np
from . import tentbasis
from .sfo_admm import SFO
from proxalgs.core import Optimizer
from proxalgs import operators
import pandas as pd

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

            # choose number of minibatches as sqrt(T)/10
            minibatch_size = np.ceil(0.1 * np.sqrt(self.num_samples)).astype('int')
            num_minibatches = int(self.num_samples / minibatch_size)

        else:
            num_minibatches = int(self.num_samples / minibatch_size)

        # slice the z-scored stimulus every tau samples, for easier dot products
        slices = _rolling_window((stimulus - np.mean(stimulus)) / np.std(stimulus), self.tau)

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

        # compute the mean firing rate
        self.meanrate = np.mean([np.mean(d['rate']) for d in self.data])

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

    def test(self):
        """
        Evaluate the model on held out data

        Relies on the model subclass having a `metrics` method, which accepts an index into the data array and returns
        a dictionary containing various metrics evaluated given the current value of the parameters (`theta`) on the
        minibatch of data at the given index

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

        >>> results, avg = model.test()

        """

        results = list()
        indices = list()
        class_key = 'set'

        # evaluate metrics on training data
        for train_idx in self.train_indices:
            results.append(self.metrics(self.data[train_idx]).update({class_key: 'train'}))
            indices.append(train_idx)

        # evaluate metrics on testing data
        for test_idx in self.test_indices:
            results.append(self.metrics(self.data[test_idx]).update({class_key: 'test'}))
            indices.append(test_idx)

        # create DataFrame to store results
        df = pd.DataFrame(data=results, index=indices)

        # compute the average over the test minibatches
        avg = dict(df.loc[df[class_key] == 'test'].mean())

        return df, avg


class LNLN(NeuralEncodingModel):
    def __init__(self, stim, rate, filter_dims, minibatch_size=None, frac_train=0.8, num_subunits=1,
                 num_tents=30, sigmasq=0.2, tent_type='gaussian', spikes=None, **kwargs):
        """
        Initializes a two layer cascade (LNLN) model

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

        Other Parameters
        ----------------
        \\*\\*kwargs : keyword arguments
            if given arguments with the keys `W` or `f`, then those values are used to initialize the filter
            or nonlinearity parameters, respectively.

        """

        # initialize the model object
        NeuralEncodingModel.__init__(self, 'lnln_exp', stim, rate, spikes, filter_dims, minibatch_size, frac_train)

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
                self.theta_init['W'][idx] = _nrm(w)

        else:

            # multiple subunits: random initialization
            if self.num_subunits > 1:
                for idx in range(self.num_subunits):
                    self.theta_init['W'][idx] = _nrm(0.1 * np.random.randn(self.stim_dim, self.tau))

            # single subunit: initialize with the STA
            else:
                self.theta_init['W'][0] = _nrm(self.sta).reshape(-1,self.sta.shape[-1])

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
                self.theta_init['f'][idx,:] = np.linalg.lstsq(self.tentparams['Phi'], nonlin_init)[0]

            self.theta_init['f'] = np.hstack((self.theta_init['f'], np.zeros((self.num_subunits,1))))

        # initialize regularizers
        self.regularizers = {'W': list(), 'f': list()}

    def f_df(self, w, f, data, param_gradient=None):
        """
        Evaluate the negative log-likelihood objective and gradient for the LNLN model class

        f, df = f_df(self, w, f, data)

        Parameters
        ----------
        w : array_like
            A numpy array containing parameter values for the first layer linear filters in the LNLN model

        f : array_like
            A numpy array containing parameter values for the first layer nonlinearity in the LNLN model

        data : dict
            Dictionary containing two keys: `stim` and `rate`, each of which is a numpy array.

        param_gradient : string (optional, default=None)
            A string indicating which parameters to compute the gradient for, either `w` or `f`

        Returns
        -------
        obj_value : float
            The negative log-likelihood objective value. Lower values indicate a better fit to the data.

        obj_gradient : array_like
            Contains the gradient (as a numpy array) with respect to the parameters given by `param_gradient`

        """

        # f is (K,P)
        # w is (K,N,tau)
        k, n, tau = w.shape
        m = (data['rate'].size - tau + 1)

        # estimate firing rate and get model response
        u, z, zgrad, logr, r = self._rate({'w': w, 'f': f}, data['stim'])

        # objective in bits (log-likelihood difference between model and mean firing rates)
        obj_value = np.mean(r - data['rate'] * logr)

        # factor in front of the gradient
        grad_factor = r - data['rate']  # dims: (M)

        # compute gradient
        if param_gradient == 'w':
            nonlin_proj = np.sum(f[:, np.newaxis, :] * zgrad, axis=2)  # dims: (K, M)
            weighted_proj = grad_factor[np.newaxis, :] * nonlin_proj  # dims: (K, M)
            obj_gradient = np.tensordot(weighted_proj, data['stim'], ([1], [1])) / float(m)

        elif param_gradient == 'f':
            obj_gradient = np.tensordot(grad_factor, z, ([0], [1])) / float(m)

        else:
            obj_gradient = None

        return obj_value, obj_gradient

    def fit(self, num_alt=2, num_steps=50, num_iter=5):
        """
        Runs an optimization algorithm to learn the parameters of the model given training data and regularizers

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        See the `proxalgs` module for more information on the optimization algorithm



        """

        # grab the initial parameters
        Wk = self.theta_init['W'].copy()
        fk = self.theta_init['f'].copy()

        # get list of training data
        train_data = [self.data[idx] for idx in self.train_indices]

        # store optimization results and parameters
        results = list()
        Ws = [Wk]
        fs = [fk]

        # runtime
        self.tinit = time.time()
        self.theta_temp = copy.deepcopy(self.theta_init)
        runtimes = list()
        evals = list()

        # not being used right now
        def callback_func(mu, res):
            if mu.ndim == 3:
                self.theta_temp['W'] = mu.copy()
            else:
                self.theta_temp['f'] = mu.copy()
            runtimes.append(time.time() - self.tinit)
            # evals.append(self.test(self.theta_temp)[1])
            self.tinit = time.time()

        # alternating optimization
        for alt_iter in range(num_alt):

            # Fit nonlinearity
            # initialize the optimizer for the nonlinearity
            def f_df_wrapper(f, d):
                return self.f_df(Wk, f, d, param_gradient='f')

            opt = Optimizer('sfo', optimizer=SFO(f_df_wrapper, fk, train_data, display=1), num_steps=num_steps)
            # TODO: FIX REGULARIZERS!
            # opt.add_regularizer()
            fk = opt.minimize(fk)
            fs.append(fk)
            results.append(opt.results)

            # Fit filters
            # initialize the optimizer for the filters
            def f_df_wrapper(W, d):
                return self.f_df(W, fk, d, param_gradient='W')

            opt = Optimizer('sfo', optimizer=SFO(f_df_wrapper, Wk, train_data, display=1), num_steps=num_steps)
            # TODO: FIX REGULARIZERS!
            # opt.add_regularizer()
            Wk = opt.minimize(Wk)
            for fi in range(Wk.shape[0]):
                Wk[fi] = _nrm(Wk[fi])   # normalize filters
            Ws.append(Wk)
            results.append(opt.results)

        # store learned parameters
        self.theta = {'W': Wk, 'f': fk}

        return results, Ws, fs, runtimes, evals

    def metrics(self, data_index):
        """
        Evaluate metrics on a given minibatch.

        Parameters
        ----------
        data_index : int
            An index into the array of minibatches (`self.data`) the evaluate

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

        # compute the model firing rate
        logr, rhat = self._rate(self.theta, self.data[data_index]['stim'])[-2:]

        # correlation coefficient
        cc = float(np.corrcoef(np.vstack((rhat, data['rate'])))[0, 1])

        # log-likelihood improvement over a mean rate model (in bits per spike)
        mu = float(np.mean(data['rate'] * np.log(self.meanrate) - self.meanrate))
        fobj = (float(np.mean(data['rate'] * logr - rhat)) - mu) / (self.meanrate * np.log(2))

        # mean squared error
        mse = float(np.mean((rhat - data['rate']) ** 2))

        return {'corrcoef': cc, 'log-likelihood improvement': fobj, 'mean squared error': mse}

    def _stim_gradient(self, stim):
        """
        Compute the model response and gradient with respect to the stimulus

        .. warning:: Work in progress

        """

        u, z, zgrad, logr, r = self._rate(self.theta, stim)

        xgrad = np.zeros_like(np.squeeze(stim))
        for idx in range(self.theta['W'].shape[0]):

            xgrad += self.theta['W'][idx] * zgrad[idx, 0, :].dot(self.theta['f'][idx, :])

        # fgrad = np.tensordot(zgrad, self.theta['f'], ([0,2],[0,1]))
        # wgrad = np.tensordot(u, self.theta['W'],([0],[0]))

        return r, xgrad * r

    def _rate(self, theta, stim):
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
        logr : array_like
        r : array_like

        """

        # filter projection
        u = np.tensordot(theta['W'], stim, ([1, 2], [0, 2]))  # dims: (K x M)

        # evaluate input at tent basis functions
        z, zgrad = tentbasis.eval_tents(u, self.tentparams)

        # compute log(rate) and the firing rate
        logr = np.tensordot(theta['f'], z, ([0, 1], [0, 2]))  # dims: (M)
        r = np.exp(logr)  # dims: (M)

        return u, z, zgrad, logr, r


def _rolling_window(a, window):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    a : array_like
        Array to add rolling window to
    window : int
        Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    """
    assert window >= 1, "`window` must be at least 1."
    assert window < a.shape[-1], "`window` is too long."

    # # with strides
    shape = a.shape[:-1] + (a.shape[-1] - window, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _nrm(x):
    """
    Normalizes data in the given array x by the (vectorized) 2-norm

    Parameters
    ----------
    x : array_like
        The input to be normalized

    Returns
    -------
    xn : array_like
        A version of the input array that has been scaled so it has a unit vectorized 2-norm

    """
    return x / np.linalg.norm(x.ravel())
