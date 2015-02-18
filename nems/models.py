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
- `LNLN` -- A subclass of `NeuralEncodingModel` that fits two layer models consisting of alternating layers of linear
    filtering and nonlinear thresholding operations. The parameters for the filter and nonlinearities of the first layer
    are learned, while the linear filter and nonlinearity of the second layer are fixed.

References
----------
Coming soon

"""

# imports
import time
import copy
from os.path import join
from functools import partial
import numpy as np
from . import tentbasis
from . import datastore
from .sfo_admm import SFO
from proxalgs import Optimizer

# exports
__all__ = ['NeuralEncodingModel', 'LNLN']


class NeuralEncodingModel(object):
    """an object which manages optimizing parameters of neural encoding models.
    Specifically, fits either LN or LNLN models given a firing rate in response
    to a spatiotemporal stimulus.
    """

    def __init__(self, modeltype, stimulus, rate, spikes, filter_dims, minibatch_size, frac_train):

        # model name / subclass
        self.modeltype = str.lower(modeltype)

        # model properties
        self.num_samples = rate.size
        self.tau = filter_dims[-1]
        self.filter_dims = filter_dims
        self.stim_dim = np.prod(filter_dims[:-1])

        # the length of the filter must be smaller than the length of the experiment
        assert self.tau <= self.num_samples, 'The filter length (in time) must be smaller than the length of the experiment.'

        # filter dimensions must be (n1 x n2 x tau), while the stimulus dimensions should be (n1*n2 x t)
        assert stimulus.shape[0] == self.stim_dim, 'Stimulus size does not match up with filter dimensions'

        ### initialize minibatches
        # set up minibatches data
        if minibatch_size is None:
            # choose number of minibatches according to the 'sweet spot' for SFO, sqrt(T)/10
            minibatch_size = np.round(10 * np.sqrt(self.num_samples)).astype('int')
            num_minibatches = int(self.num_samples / minibatch_size)
        else:
            num_minibatches = int(self.num_samples / minibatch_size)

        ### initialize data
        # slice the z-scored stimulus every tau samples
        slices = _rolling_window((stimulus-np.mean(stimulus))/np.std(stimulus), self.tau)

        # store stimulus and rate data for each minibatch in a list
        self.data = list()
        for idx in range(num_minibatches):

            # indices for this minibatch
            minibatch_indices = slice(idx * minibatch_size, (idx + 1) * minibatch_size)

            # z-score the stimulus and save each minibatch, along with the rate
            self.data.append({
                'stim': slices[:, minibatch_indices, :],
                'rate': rate[minibatch_indices],
                'spikes': np.where(spikes[minibatch_indices] > 0)[0]
            })

        # reproducible experiments
        self.random_seed = 1234
        np.random.seed(self.random_seed)

        # split up data into train/validation/test sets
        num_train = int(np.round(frac_train * num_minibatches))
        num_validation = int(np.ceil(0.5*(num_minibatches - num_train)))
        indices = np.arange(num_minibatches)
        np.random.shuffle(indices)
        self.train_indices = indices[:num_train]
        self.validation_indices = indices[num_train:(num_train+num_validation)]
        self.test_indices = indices[(num_train+num_validation):]

        # compute the STA
        self._getsta()

        # compute the mean firing rate
        self.meanrate = np.mean([np.mean(d['rate']) for d in self.data])

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def __str__(self):
        return "Neural encoding model, " + self.modeltype

    def _getsta(self):
        """Compute an STA
        """
        T = float(self.data[0]['rate'].size)
        stas = [np.tensordot(d['stim'], d['rate'], ([1], [0]))/T for d in self.data]
        self.sta = np.mean(stas,axis=0).reshape(self.filter_dims)


    def add_regularizer(self, theta_key, proxfun, **kwargs):
        """Add a proximal operator / objective to optimize

        Arguments
        ---------
        proxfun     -- proxfun is a function that evaluates a proximal operator (see proxops.py for some examples)
                        It takes as input the current parameter values, the parameter rho (momentum term), and
                        optional keyword args

        """
        assert 'regularizers' in self.__dict__, 'List of regularizers not initialized!'
        assert hasattr(proxops, proxfun), 'Could not find function ' + proxfun + ' in proxops.py'

        def wrapper(v, rho, **kwargs):

            # copy the parameters
            v_new = copy.deepcopy(v)

            # apply the proximal operator to each element in
            for idx, param in enumerate(v):
                v_new[idx] = getattr(proxops, proxfun)(param.copy(), float(rho), **kwargs)

            return v_new

        # add proximal operator to the list
        self.regularizers[theta_key].append(partial(wrapper, **kwargs))

    def notify(self, msg, client, desc=''):
        titlestr = '%s: %s (%s)' % (self.modeltype, desc, time.strftime('%h %d %I:%M %p'))
        client.send_message(msg, title=titlestr)

    def init_datastore(self, name, desc, extra_headers):
        main_headers = ('Corr. coeff. (train)', 'Objective (train)', 'Mean squared error (train)',
                        'Corr. coeff. (test)', 'Objective (test)', 'Mean squared error (test)')
        self.db = datastore.Datastore(name, desc, main_headers + extra_headers)

    def fit_final_nonlinearity(self, bin_range_sigma=5, num_bins=2000, num_tents=100):
        """
        Fit the final nonlinearity

        """

        # estimate range of the input to the final nonlinearity
        logr = np.hstack([self._rate(self.theta, d['stim'])[-2] for d in self.data])
        bin_range = (np.mean(logr) - bin_range_sigma*np.std(logr), np.mean(logr) + bin_range_sigma*np.std(logr))
        sigmasq = np.squeeze(0.1 * np.diff(bin_range) / float(num_tents))

        # tent basis functions for the final nonlinearity
        tentparams = tentbasis.build_tents(num_bins, bin_range, num_tents, tent_type='linear', sigmasq=sigmasq)

        # build the objective function
        def f_df(theta, d):
            logr = self._rate(theta, d)[-2]
            err, err_grad = huber(rhat - d['rate'], delta=10.0)
            z, zgrad = tentbasis.eval_tents(u, self.tentparams)


        # def f_df()

    def test(self, theta, metadata=()):
        """
        Compare model to true (held out) data
        results = model.test(theta_key='theta_sfo', metadata=())
        returns a list of tuples, each tuple is of the following form:
        (metadata, minibatch_idx, set, cc, fobj, mse)
        where minibatch_idx is the index of the minibatch,
        set is either 'train' or 'test'
        and cc (correlation coefficient), fobj (main objective), and mse (mean squared error) are numbers
        """
        results = list()
        train_res = list()
        validation_res = list()
        test_res = list()

        for train_idx in self.train_indices:
            tr = self.test_metrics(theta, self.data[train_idx])[0]
            results.append(metadata + (train_idx, 'train') + tr)
            train_res.append(tr)

        for validation_idx in self.validation_indices:
            tr = self.test_metrics(theta, self.data[validation_idx])[0]
            results.append(metadata + (validation_idx, 'validation') + tr)
            validation_res.append(tr)

        for test_idx in self.test_indices:
            tr = self.test_metrics(theta, self.data[test_idx])[0]
            results.append(metadata + (test_idx, 'test') + tr)
            test_res.append(tr)

        num_train = float(len(self.train_indices))
        num_validation = float(len(self.validation_indices))
        num_test = float(len(self.test_indices))
        avg = tuple(np.nanmean(np.vstack(train_res),      axis=0)) + \
              tuple(np.nanmean(np.vstack(validation_res), axis=0)) + \
              tuple(np.nanmean(np.vstack(test_res),       axis=0))
        spread = tuple(np.nanstd(np.vstack(train_res),      axis=0) / np.sqrt(num_train)) + \
                 tuple(np.nanstd(np.vstack(validation_res), axis=0) / np.sqrt(num_validation)) + \
                 tuple(np.nanstd(np.vstack(test_res),       axis=0) / np.sqrt(num_test))

        return results, avg, spread

    def plot(self, filename=None):

        # determine number of subunits
        nsub = self.theta['W'].shape[0]

        # make a figure
        fig = plt.figure(figsize=(6, 4 * nsub))
        fig.clf()
        sns.set_style('whitegrid')

        # build axes for each subunit
        for idx in range(nsub):

            # top row: filter
            ax = plt.subplot2grid((3,nsub), (0,idx), rowspan=2)
            # viz.plotsta1D(self.theta['W'][idx], ax)
            W = self.theta['W'][idx] - np.median(self.theta['W'][idx])
            ax.imshow(W, cmap='seismic', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
            ax.set_title('Subunit #%i' % (idx+1), fontsize=22)

            # bottom row: nonlinearities
            ax = plt.subplot2grid((3,nsub), (2,idx))
            nonlin_func = np.exp(self.tentparams['Phi'].dot(self.theta['f'][idx, :].T))

            # histogram
            u = np.hstack([np.tensordot(d['stim'], self.theta['W'][idx], ([0, 2], [0, 1])) for d in self.data])
            counts, bin_edges = np.histogram(u, bins=50, range=self.tentparams['tent_span'])
            centers = bin_edges[:-1] + 0.5*np.diff(bin_edges)
            count = counts.astype('float') / float(np.max(counts))
            print(centers)
            print(count)
            ax.plot(centers, count * np.max(nonlin_func), '-', color='gray')
            ax.fill(centers, count * np.max(nonlin_func), color='lightgray', alpha=0.15)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # the nonlinearity
            ax.plot(self.tentparams['tent_x'], nonlin_func, '-', color='lightcoral', linewidth=3)
            ax.set_xlim(self.tentparams['tent_span'][0]+0.1, self.tentparams['tent_span'][1]-0.1)

        if filename is None:
            plt.show()
            plt.draw()

        else:
            plt.savefig(join(filename, 'cell' + str(self.cellidx) + '.png'))
            plt.close()
            del fig


class LNLN(NeuralEncodingModel):
    def __init__(self, stim, rate, filter_dims, minibatch_size=None, frac_train=0.8, num_subunits=1,
                 num_tents=30, sigmasq=0.2, tent_type='gaussian', spikes=None, **kwargs):
        """
        Initializes a two layer cascade (LNLN) model

        Usage
        -----
        >> model = LNLN(stim, rate, filter_dims, theta_init=None, minibatchSize=None)

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
            roughly equal to 0.1 * sqrt(T), the SFO 'sweet spot'

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

        Optional Arguments
        ------------------
        optionally pass in initial parameter values, W=W_init and f=f_init

        """

        ### initialize the model object
        NeuralEncodingModel.__init__(self, 'lnln_exp', stim, rate, spikes, filter_dims, minibatch_size, frac_train)

        # initialize model parameters
        # default # of subunits
        if 'W' in kwargs:
            self.num_subunits = kwargs['W'].shape[0]
        else:
            self.num_subunits = num_subunits

        # initialize tent basis functions
        num_tent_samples = 1000
        tent_span = (-5,5)          # works for z-scored input
        self.tentparams = tentbasis.build_tents(num_tent_samples, tent_span, num_tents, tent_type=tent_type, sigmasq=sigmasq)

        # initialize filter parameters
        self.theta_init = dict()
        self.theta_init['W'] = np.zeros((self.num_subunits,) + (self.stim_dim, self.tau))
        self.theta_init['f'] = np.zeros((self.num_subunits, self.tentparams['num_tents']))

        if 'W' in kwargs:
            # normalize each subunit
            for idx, w in enumerate(kwargs['W']):
                self.theta_init['W'][idx] = _nrm(w)
        else:
            # multiple subunits: random init?
            if self.num_subunits > 1:
                for idx in range(self.num_subunits):
                    self.theta_init['W'][idx] = _nrm(0.1 * np.random.randn(self.stim_dim, self.tau))

            else:
                self.theta_init['W'][0] = _nrm(self.sta).reshape(-1,self.sta.shape[-1])

        # initialize nonlinearity parameters
        if 'f' in kwargs:
            self.theta_init['f'] = kwargs['f']
        else:
            for idx in range(self.num_subunits):
                # initialize to a linear function
                ts = self.tentparams['tent_span']
                nonlin_init = np.linspace(ts[0], ts[1], self.tentparams['num_tent_samples'])
                self.theta_init['f'][idx,:] = np.linalg.lstsq(self.tentparams['Phi'], nonlin_init)[0]

            self.theta_init['f'] = np.hstack((self.theta_init['f'], np.zeros((self.num_subunits,1))))

        # initialize regularizers
        self.regularizers = {'W': list(), 'f': list()}


    def f_df(self, W, f, data, param_gradient=None):
        """
        Evaluate the negative log-likelihood objective and gradient for the LNLN model class

        f, df = f_df(self, W, f, data)

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
        u, z, zgrad, logr, r = self._rate({'W': W, 'f': f}, data['stim'])

        # objective in bits (log-likelihood difference between model and mean firing rates)
        obj_value = np.mean(r - data['rate'] * logr)

        # factor in front of the gradient
        grad_factor = r - data['rate']  # dims: (M)

        # compute gradient
        if param_gradient=='W':
            nonlin_proj = np.sum(f[:, np.newaxis, :] * zgrad, axis=2)  # dims: (K, M)
            weighted_proj = grad_factor[np.newaxis, :] * nonlin_proj  # dims: (K, M)
            obj_gradient = np.tensordot(weighted_proj, data['stim'], ([1], [1])) / float(m)

        elif param_gradient == 'f':
            obj_gradient = np.tensordot(grad_factor, z, ([0], [1])) / float(m)

        else:
            obj_gradient = None

        return obj_value, obj_gradient


    def fit(self, num_alt=2, num_steps=50, num_iter=5):

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


    def test_metrics(self, theta, data):
        """evaluate test metrics at given parameters

        res, rhat = test_metrics(self, theta, data)

        :param theta:
            dictionary of parameters to test

        :param data:
            dictionary of data parameters for a minibatch, data['stim'] and data['rate']

        :return:
            res: a tuple of metrics between the model and true firing rate.
                 (correlation coefficient, neg. log likelihood objective, mean squared error)
            rhat: the predicted model rate

        """

        logr, rhat = self._rate(theta, data['stim'])[-2:]

        # correlation coefficient
        cc = float(np.corrcoef(np.vstack((rhat, data['rate'])))[0, 1])

        # relative log-likelihood, difference from mean rate model (bits per spike)
        mu = float(np.mean(data['rate'] * np.log(self.meanrate) - self.meanrate))
        fobj = (float(np.mean(data['rate'] * logr - rhat)) - mu) / (self.meanrate * np.log(2))

        # mean squared error
        mse = float(np.mean((rhat - data['rate']) ** 2))

        return (cc, fobj, mse), rhat

    def _stim_gradient(self, stim):
        """
        Compute the model response and gradient with respect to the stimulus
        """

        u, z, zgrad, logr, r = self._rate(self.theta, stim)

        xgrad = np.zeros_like(np.squeeze(stim))
        for idx in range(self.theta['W'].shape[0]):

            xgrad += self.theta['W'][idx] * zgrad[idx, 0, :].dot(self.theta['f'][idx, :])

        # fgrad = np.tensordot(zgrad, self.theta['f'], ([0,2],[0,1]))
        # wgrad = np.tensordot(u, self.theta['W'],([0],[0]))

        return r, xgrad*r

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
    >> x=np.arange(10).reshape((2,5))
    >> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:
    >> np.mean(rolling_window(x, 3), -1)
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
    Normalizes data in the given array x by the (vectorized) norm

    """
    return x / np.linalg.norm(x.ravel())
