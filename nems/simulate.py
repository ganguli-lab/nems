"""
A module for generating data from simulated models

.. warning:: Work in progress

"""

def simulateexpt(self, numBatches=100, numSamplesPerBatch=1000):
    """Simulates data from the model
    """
    # TODO: update simulateexpt to work with the new objective f_df

    # generate data
    self.data = list()
    for j in range(numBatches):

        # generate stimuli
        stim = np.random.randn(numSamplesPerBatch, self.stim_dim)

        # generate 'true' firing rate
        rate = self.modelrate(self.theta_true, {'stim': stim})[1]

        # hack: cap rate to a reasonable amount
        if np.any(rate * self.dt > 100):
            print('[warning] Model simulation has very high firing rates. Maybe tweak parameters? Capping to 100')
            rate[rate * self.dt > 100] = 100.0 / self.dt

        # generate spike counts
        spkcount = np.random.poisson(rate * self.dt)

        # store
        self.data.append({'stim': stim, 'rate': rate, 'spkcount': spkcount})

    # store dimensions
    self.num_samples = numSamplesPerBatch

    # data has been loaded
    self.loaded = True


def simulate_LNLN(self, dim=100, num_tents=30, num_tent_samples=1000, sigmasq=0.1, dt=1, numBatches=100,
             numSamplesPerBatch=1000):

    # # generate fake model parameters
    self.dt = dt

    # subunits
    self.num_subunits = 3
    self.stim_dim = dim

    # parameters
    numBases = 100
    surround_scaling = 0.2
    phix = np.linspace(-1, 1, dim)

    # build center and surround Gaussians
    centers = tentbasis.makeGaussianBasis(phix, numBases, sigmasq=0.005)[0]
    surrounds = tentbasis.makeGaussianBasis(phix, numBases, sigmasq=0.05)[0]

    # subtract to form receptive fields (rfs)
    rfs = centers - surround_scaling * surrounds

    # pick receptive fields
    W = np.zeros((dim, self.num_subunits))
    W[:, 0] = rfs[:, 30]
    W[:, 1] = rfs[:, 50]
    W[:, 2] = rfs[:, 70]

    # initialize filter parameters
    self.theta_true = dict()

    # subunit filters
    self.theta_true['W'] = nrm(W)

    # subunit nonlinearities
    self.num_tents = num_tents
    self.num_tent_samples = num_tent_samples
    self.sigmasq = sigmasq  # np.squeeze(sigmasq * np.diff(self.tent_span))
    self.init_tents()

    offsets = 1.5 + 0.0 * np.random.randn(self.num_subunits, 1)
    slopes = 2.5 + 0.0 * np.random.rand(self.num_subunits, 1)
    y = np.zeros((self.tent_x.size, self.num_subunits))
    for j in range(self.num_subunits):
        y[:, j] = slopes[j] * np.log(1 + np.exp(1.5 * self.tent_x - offsets[j])) - 5
    self.theta_true['f'] = np.linalg.lstsq(self.Phi, y)[0]

    # # simulate data
    self.simulatedata(numBatches=numBatches, numSamplesPerBatch=numSamplesPerBatch)
