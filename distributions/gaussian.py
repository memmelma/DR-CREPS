import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

import mushroom_rl

class GaussianDiagonalDistribution(mushroom_rl.distributions.distribution.Distribution):
    """
    Gaussian distribution with diagonal covariance matrix. The parameters
    vector represents the mean and the standard deviation for each dimension.
    Implements Prioritized Exploration.
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

        self._lambd = 0
        self._sample_strat = None
        self._allowed_sample_strats = ['fixed', 'percentage', 'importance', 'PRO', 'None']
        
        self._top_k = []
        self._importance = None

        self._add_save_attr(
            _mu='numpy',
            _std='numpy'
        )

    def set_sample_strat(self, sample_strat, lambd):
        assert sample_strat in self._allowed_sample_strats, f"'sample_strat={sample_strat}' not supported! Choose one of {self._allowed_sample_strats}!"
        assert 0 <= lambd and lambd <= 1, f'lambd must be 0 \leq \lambda \leq 1 got {lambd}'
        self._sample_strat = sample_strat
        self._lambd = lambd

    def sample(self):
        from copy import copy
        std_tmp = copy(self._std**2)

        selection = np.in1d(range(std_tmp.shape[0]), self._top_k)

        if self._sample_strat == 'fixed' and len(self._top_k):
            std_tmp[~selection] = self._lambd
        elif self._sample_strat == 'percentage' and len(self._top_k):
            std_tmp[~selection] = std_tmp[~selection] * self._lambd
        elif self._sample_strat == 'importance' and self._importance is not None:
            std_tmp = std_tmp * (self._importance+1e-18)
        elif self._sample_strat == 'PRO' and self._importance is not None:
            std_tmp = std_tmp * self._importance

        sigma = np.diag(std_tmp)

        return np.random.multivariate_normal(self._mu, sigma)
        
    def update_importance(self, importance):
        self._importance = importance
        if self._sample_strat == 'PRO':
            self._std = ( 1 - self._importance ) * self.previous_std + self._importance * self._std

    def sample_standard(self):
        sigma = np.diag(self._std**2)
        return np.random.multivariate_normal(self._mu, sigma)

    def log_pdf(self, theta):
        sigma = np.diag(self._std ** 2)
        return multivariate_normal.logpdf(theta, self._mu, sigma)

    def __call__(self, theta):
        sigma = np.diag(self._std ** 2)
        return multivariate_normal.pdf(theta, self._mu, sigma)

    def entropy(self):
        return 0.5 * np.log(np.product(2*np.pi*np.e*self._std**2))

    def mle(self, theta, weights=None, indices=[]):
        
        self._top_k = indices
        self.previous_std = np.copy(self._std)

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

    def con_wmle(self, theta, weights, eps, kappa, indices=[]):
        
        self._top_k = indices
        self.previous_std = np.copy(self._std)
        
        n_dims = len(self._mu)
        mu = self._mu
        sigma = self._std

        eta_omg_opt_start = np.array([1000., 0.])
        res = minimize(mushroom_rl.distributions.GaussianDiagonalDistribution._lagrangian_eta_omega, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa),
                       method='L-BFGS-B')

        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = mushroom_rl.distributions.GaussianDiagonalDistribution._compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta_opt, omg_opt)

        self._mu, self._std = mu_new, sigma_new

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

class GaussianDistributionGDR(mushroom_rl.distributions.distribution.Distribution):
    """
    Gaussian distribution with full covariance matrix. The parameters
    vector represents the mean and the current diagonal matrix containing
    the singular values of the current covariance matrix obtained by SVD.
    Implements Prioritized Exploration.
    """
    def __init__(self, mu, sigma):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            sigma (np.ndarray): initial covariance matrix of the distribution.

        """
        self._mu = mu
        self._u, self._s, self._vh = scipy.linalg.svd(sigma, lapack_driver='gesvd')
        self.sigma_prime= np.diag(self._s)

        self._lambd = 0
        self._sample_strat = None
        self._allowed_sample_strats = ['fixed', 'percentage', 'importance', 'None']
        
        self._top_k = []
        self._importance = None

        self._add_save_attr(
            _mu='numpy',
            _s='numpy'
        )
    
    def set_sample_strat(self, sample_strat, lambd):
        assert sample_strat in self._allowed_sample_strats, f"'sample_strat={sample_strat}' not supported! Choose one of {self._allowed_sample_strats}!"
        assert 0 <= lambd and lambd <= 1, f'lambd must be 0 \leq \lambda \leq 1 got {lambd}'
        self._sample_strat = sample_strat
        self._lambd = lambd

    def sample(self):
        return np.random.multivariate_normal(self._mu, self.sigma_prime)

    def get_parameters(self):
        rho = np.empty(self.parameters_size)
        n_dims = len(self._mu)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._s.flatten()

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        self._mu = rho[:n_dims]
        sigma = rho[n_dims:].reshape(n_dims, n_dims)
        self._u, self._s, self._vh = scipy.linalg.svd(sigma, lapack_driver='gesvd')
    
    def mle_gdr(self, theta, indices=None, weights=None):
        self._top_k = indices

        # get projected mu, Sigma
        mu = np.zeros_like(self._mu)
        sigma_old = np.copy(np.diag(self._s))

        # apply GDR via partial REPS update
        mu = mu[indices]
        theta = theta[:, indices]

        sumD = np.sum(weights)
        sumD2 = np.sum(weights**2)
        Z = sumD - sumD2 / sumD

        mu_new = ( weights.dot(theta) / sumD )

        delta = theta - mu_new

        sigma_new = delta.T.dot(np.diag(weights)).dot(delta) / Z

        # substitute back mu
        mu_new_tmp = np.zeros_like(self._mu)
        mu_new_tmp[indices] = mu_new
        mu_new = mu_new_tmp
        
        # apply PE
        if self._sample_strat == 'percentage':
            sigma_prime_new_tmp = sigma_old * self._lambd
        else:
            sigma_prime_new_tmp = sigma_old
        
        # substitute back S' to Sigma'
        sigma_prime_new_tmp[indices,indices] = np.diag(sigma_new)
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i != j:
                    sigma_prime_new_tmp[idx_i,idx_j] = sigma_new[i,j]
        self.sigma_prime= self._u @ sigma_prime_new_tmp @ self._vh

        # substitute back S to Sigma
        sigma_new_tmp = sigma_old
        sigma_new_tmp[indices,indices] = np.diag(sigma_new)
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i != j:
                    sigma_new_tmp[idx_i,idx_j] = sigma_new[i,j]
        sigma_new = self._u @ sigma_new_tmp @ self._vh

        # set new parameters
        self._mu = self._mu + self._u @ mu_new
        self._u, self._s, self._vh = scipy.linalg.svd(sigma_new, lapack_driver='gesvd')

    def con_wmle_gdr(self, theta, weights, eps, kappa, indices):
        self._top_k = indices

        # get projected mu, Sigma
        mu = np.zeros_like(self._mu)
        sigma_old = np.copy(np.diag(self._s))

        # apply GDR via partial CREPS update
        mu = mu[indices]
        sigma = np.diag(np.diag(sigma_old)[indices])
        n_dims = len(mu)
        theta = theta[:, indices]

        eta_omg_opt_start = np.array([1000., 0.])
        res = minimize(mushroom_rl.distributions.GaussianCholeskyDistribution._lagrangian_eta_omega, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa),
                       method='L-BFGS-B')

        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = mushroom_rl.distributions.GaussianCholeskyDistribution._compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta_opt, omg_opt)

        # substitute back mu
        mu_new_tmp = np.zeros_like(self._mu)
        mu_new_tmp[indices] = mu_new
        mu_new = mu_new_tmp

        # apply PE
        if self._sample_strat == 'percentage':
            sigma_prime_new_tmp = sigma_old * self._lambd
        else:
            sigma_prime_new_tmp = sigma_old

        # substitute back S' to Sigma'
        sigma_prime_new_tmp[indices,indices] = np.diag(sigma_new)
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i != j:
                    sigma_prime_new_tmp[idx_i,idx_j] = sigma_new[i,j]
        self.sigma_prime= self._u @ sigma_prime_new_tmp @ self._vh
        
        # substitute back S to Sigma
        sigma_new_tmp = np.copy(sigma_old)
        sigma_new_tmp[indices,indices] = np.diag(sigma_new)
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i != j:
                    sigma_new_tmp[idx_i,idx_j] = sigma_new[i,j]
        sigma_new = self._u @ sigma_new_tmp @ self._vh

        # set new parameters
        self._mu = self._mu + self._u @ mu_new
        self._u, self._s, self._vh = scipy.linalg.svd(sigma_new, lapack_driver='gesvd')

    def entropy(self):
        sigma = self._u @ np.diag(self._s) @ self._vh
        std = np.sqrt(np.diag(sigma))
        return 0.5 * np.log(np.product(2*np.pi*np.e*std**2))

    @property
    def parameters_size(self):
        n_dims = len(self._mu)

        return 2 * n_dims

from mushroom_rl.distributions import GaussianCholeskyDistribution

class GaussianCholeskyDistribution_(GaussianCholeskyDistribution):
    
    def con_wmle(self, theta, weights, eps, kappa):
        n_dims = len(self._mu)
        mu =self._mu
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        
        eta_omg_start = np.array([1000, 0])
        res = minimize(GaussianCholeskyDistribution._lagrangian_eta_omega, eta_omg_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa),
                       method='L-BFGS-B')
        
        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianCholeskyDistribution._compute_mu_sigma_from_lagrangian(weights, theta, mu, sigma, eta_opt, omg_opt)

        self._mu, self._chol_sigma = mu_new, np.linalg.cholesky(sigma_new)