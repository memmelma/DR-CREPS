import numpy as np
from mushroom_rl.distributions import Distribution
from scipy.stats import multivariate_normal

from copy import copy

class GaussianDiagonalDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix. The parameters
    vector represents the mean and the standard deviation for each dimension.

    """
    def __init__(self, mu, std):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            std (np.ndarray): initial vector of standard deviations for each
                variable of the distribution.

        """
        assert(len(std.shape) == 1)
        self._mu = mu
        self._std = std

        self._sample_type = None
        self._top_k = [] 
        self._eps = 0
        self._kappa = 0

        self._add_save_attr(
            _mu='numpy',
            _std='numpy'
        )

    def set_fixed_sample(self, eps=1e-10):
        self._eps = eps
        self._sample_type = 'fixed'
    
    def set_percentage_sample(self, kappa=0.1):
        self._kappa = kappa
        self._sample_type = 'percentage'

    def sample(self):
        std_tmp = copy(self._std)

        if self._sample_type == 'fixed':
            std_tmp[self._top_k] = self._eps
        elif self._sample_type == 'percentage':
            std_tmp[self._top_k] = std_tmp[self._top_k] * self._kappa
            print(std_tmp[self._top_k])

        sigma = np.diag(std_tmp**2)
        return np.random.multivariate_normal(self._mu, sigma)

    def log_pdf(self, theta):
        sigma = np.diag(self._std ** 2)
        return multivariate_normal.logpdf(theta, self._mu, sigma)

    def __call__(self, theta):
        sigma = np.diag(self._std ** 2)
        return multivariate_normal.pdf(theta, self._mu, sigma)

    def entropy(self):
        return 0.5 * np.log(np.product(2*np.pi*np.e*self._std**2))

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
            self._std = np.std(theta, axis=0)
        else:
            sumD = np.sum(weights)
            sumD2 = np.sum(weights**2)
            Z = sumD - sumD2 / sumD

            self._mu = weights.dot(theta) / sumD

            delta2 = (theta - self._mu)**2
            self._std = np.sqrt(weights.dot(delta2) / Z)

    def diff_log(self, theta):
        n_dims = len(self._mu)

        sigma = self._std**2

        g = np.empty(self.parameters_size)

        delta = theta - self._mu

        g_mean = delta / sigma
        g_std = delta**2 / (self._std**3) - 1 / self._std

        g[:n_dims] = g_mean
        g[n_dims:] = g_std
        return g

    def get_parameters(self):
        rho = np.empty(self.parameters_size)
        n_dims = len(self._mu)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._std

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        self._mu = rho[:n_dims]
        self._std = rho[n_dims:]

    @property
    def parameters_size(self):
        return 2 * len(self._mu)
