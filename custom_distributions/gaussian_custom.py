import numpy as np
from .distribution import Distribution
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

class GaussianDistribution(Distribution):
    """
    Gaussian distribution with fixed covariance matrix. The parameters
    vector represents only the mean.

    """
    def __init__(self, mu, sigma):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            sigma (np.ndarray): covariance matrix of the distribution.

        """
        self._mu = mu
        self._sigma = sigma
        self._inv_sigma = np.linalg.inv(sigma)

        self._add_save_attr(
            _mu='numpy',
            _sigma='numpy',
            _inv_sigma='numpy'
        )

    def sample(self):
        return np.random.multivariate_normal(self._mu, self._sigma)

    def log_pdf(self, theta):
        return multivariate_normal.logpdf(theta, self._mu, self._sigma)

    def __call__(self, theta):
        return multivariate_normal.pdf(theta, self._mu, self._sigma)

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(2*np.pi*np.e*self._sigma))

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
        else:
            self._mu = weights.dot(theta) / np.sum(weights)

    def diff_log(self, theta):
        delta = theta - self._mu
        g = self._inv_sigma.dot(delta)

        return g

    def get_parameters(self):
        return self._mu

    def set_parameters(self, rho):
        self._mu = rho

    @property
    def parameters_size(self):
        return len(self._mu)


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

        self._gamma = 0
        self._sample_type = None
        
        self._top_k = []
        self._importance = None

        self._add_save_attr(
            _mu='numpy',
            _std='numpy'
        )

    def set_fixed_sample(self, gamma=1e-10):
        self._gamma = gamma
        self._sample_type = 'fixed'

    def set_percentage_sample(self, gamma=0.1):
        self._gamma = gamma
        self._sample_type = 'percentage'

    def set_importance_sample(self):
        self._sample_type = 'importance'
    
    def set_PRO_sample(self):
        self._sample_type = 'PRO'

    def sample(self):
        from copy import copy
        std_tmp = copy(self._std**2)

        selection = np.in1d(range(std_tmp.shape[0]), self._top_k)

        if self._sample_type == 'fixed' and len(self._top_k):
            std_tmp[~selection] = self._gamma
        elif self._sample_type == 'percentage' and len(self._top_k):
            std_tmp[~selection] = std_tmp[~selection] * self._gamma
        elif self._sample_type == 'importance' and self._importance is not None:
            std_tmp = std_tmp * (self._importance+1e-18)
        elif self._sample_type == 'PRO' and self._importance is not None:
            std_tmp = std_tmp * self._importance
            # std_tmp = std_tmp

        sigma = np.diag(std_tmp)
        try:
            return np.random.multivariate_normal(self._mu, sigma)
        except ValueError:
            print(self._importance)
            print(sigma)
            exit()

    def update_importance(self, importance):
        self._importance = importance
        if self._sample_type == 'PRO':
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

    def mle(self, theta, weights=None):
        
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

        eta_omg_opt_start = np.array([1., 1.])
        res = minimize(GaussianDiagonalDistribution._lagrangian_eta_omg, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa))

        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianDiagonalDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta_opt, omg_opt, kappa)

        self._mu, self._std = mu_new, sigma_new

        #TODO: remove
        from tqdm import tqdm
        sigma = np.diag(sigma**2)
        sigma_new = np.diag(sigma_new**2)
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)
        kl = GaussianDiagonalDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims)
        entropy = GaussianDiagonalDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - GaussianDiagonalDistribution._closed_form_entropy(logdet_sigma, n_dims)
        
        # tqdm.write(f'\nKL constraint: KL: {kl:2.6f} eps: {eps:2.6f}')

        return kl, entropy, self._mu

    def con_wmle_mi(self, theta, weights, eps, kappa, indices):
        
        self.previous_std = np.copy(self._std)

        self._top_k = indices

        n_dims = len(indices)
        mu = self._mu[indices]
        sigma = self._std[indices]
        theta = theta[:, indices] 

        eta_omg_opt_start = np.array([1., 1.])
        res = minimize(GaussianDiagonalDistribution._lagrangian_eta_omg, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa))

        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianDiagonalDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta_opt, omg_opt, kappa)

        from copy import copy
        mu = copy(self._mu)
        sigma = copy(self._std)

        self._mu[indices], self._std[indices] = mu_new, sigma_new

        mu_new = self._mu
        sigma_new = self._std
        n_dims = len(self._mu)

        from tqdm import tqdm
        sigma = np.diag(sigma**2)
        sigma_new = np.diag(sigma_new**2)
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)
        kl = GaussianDiagonalDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims)
        entropy = GaussianDiagonalDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - GaussianDiagonalDistribution._closed_form_entropy(logdet_sigma, n_dims)
        
        # tqdm.write(f'\nKL constraint: KL: {kl:2.6f} eps: {eps:2.6f}')

        return kl, entropy, self._mu

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

    @staticmethod
    def closed_form_mu1_sigma_new(*args):
        weights, theta, mu, sigma, n, eps, eta, omg, kappa = args
        weights_sum = np.sum(weights)

        mu_new = (weights @ theta + eta * mu) / (weights_sum + eta)
        sigma_new =  ( np.sum([w_i * (theta_i-mu_new)**2 for theta_i, w_i in zip(theta, weights)], axis=0) + eta*sigma**2 + eta*(mu_new - mu)**2 ) / ( weights_sum + eta - omg )
        sigma_new = np.sqrt(sigma_new)
        return mu_new, sigma_new

    @staticmethod
    def _closed_form_KL_constraint_M_projection(*args):
        mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims = args
        
        return 0.5*(np.trace(sigma_new_inv@sigma) - n_dims + logdet_sigma_new - logdet_sigma + (mu_new - mu).T @ sigma_new_inv @ (mu_new - mu))
    
    @staticmethod
    def _closed_form_entropy(*args):
        logdet_sigma, n_dims = args
        c = n_dims * np.log(2*np.pi)
        
        return 0.5 * (logdet_sigma + c + n_dims)

    @staticmethod
    def _lagrangian_eta_omg(lag_array, *args):
        weights, theta, mu, sigma, n_dims, eps, kappa = args
        eta, omg = lag_array[0], lag_array[1]

        mu_new, sigma_new = GaussianDiagonalDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta, omg, kappa)
        
        sigma = np.diag(sigma**2)
        sigma_new = np.diag(sigma_new**2)

        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)

        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)

        c = n_dims * np.log(2*np.pi)

        sum1 = np.sum([w_i * (-0.5*(theta_i - mu_new)[:,np.newaxis].T @ sigma_new_inv @ (theta_i -  mu_new)[:,np.newaxis] - 0.5 * logdet_sigma_new - c/2) for w_i, theta_i in zip(weights, theta)])
        
        sum2 = eta * (eps - GaussianDiagonalDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims))
        
        sum3 = omg * (GaussianDiagonalDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - ( GaussianDiagonalDistribution._closed_form_entropy(logdet_sigma, n_dims) - kappa ) )

        return sum1 + sum2 + sum3

class GaussianDistributionMI(Distribution):
    """
    Gaussian distribution with full covariance matrix. The parameters
    vector represents the mean and the Cholesky decomposition of the
    covariance matrix. This parametrization enforce the covariance matrix to be
    positive definite.

    """
    def __init__(self, mu, sigma):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            sigma (np.ndarray): initial covariance matrix of the distribution.

        """
        self._mu = mu
        self._u, self._s, self._vh = np.linalg.svd(sigma)

        self._gamma = 0
        self._sample_type = None
        
        self._top_k = []
        self._importance = None

        self._add_save_attr(
            _mu='numpy',
            _chol_sigma='numpy'
        )

    def set_fixed_sample(self, gamma=1e-10):
        self._gamma = gamma
        self._sample_type = 'fixed'

    def set_percentage_sample(self, gamma=0.1):
        self._gamma = gamma
        self._sample_type = 'percentage'

    def set_importance_sample(self):
        self._sample_type = 'importance'
    
    def sample(self):
        from copy import copy
        std_tmp = copy(self._s)

        selection = np.in1d(range(std_tmp.shape[0]), self._top_k)

        if self._sample_type == 'fixed' and len(self._top_k):
            std_tmp[~selection] = self._gamma
        elif self._sample_type == 'percentage' and len(self._top_k):
            std_tmp[~selection] = std_tmp[~selection] * self._gamma
        elif self._sample_type == 'importance' and self._importance is not None:
            std_tmp = std_tmp * self._importance

        sigma = np.diag(std_tmp)

        theta = np.random.multivariate_normal(np.zeros_like(self._mu), sigma)[:, np.newaxis]
        return ( self._u @ theta + self._mu[:,np.newaxis] ).squeeze()

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
        self._u, self._s, self._vh = np.linalg.svd(sigma)
    
    def mle_mi_diag(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
            sigma = np.cov(theta, rowvar=False)
        else:
            sumD = np.sum(weights)
            sumD2 = np.sum(weights**2)
            Z = sumD - sumD2 / sumD

            self._mu = weights.dot(theta) / sumD

            delta = theta - self._mu

            sigma = delta.T.dot(np.diag(weights)).dot(delta) / Z

        self._u, self._s, self._vh = np.linalg.svd(sigma)

    def mle_mi_full(self, theta, indices, weights=None):
        
        self._top_k = indices

        mu = np.zeros_like(self._mu)
        mu_old = np.copy(self._mu)
        sigma = np.diag(self._s)
        sigma_old = np.copy(np.diag(self._s))

        mu = mu[indices]
        sigma = np.diag(np.diag(sigma_old)[indices])
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i is not j:
                    sigma[i,j] = sigma_old[idx_i,idx_j]

        theta = theta[:, indices]

        # update
        sumD = np.sum(weights)
        sumD2 = np.sum(weights**2)
        Z = sumD - sumD2 / sumD

        # self._mu = weights.dot(theta) / sumD
        mu_new = ( weights.dot(theta) / sumD )

        # delta = theta - self._mu
        delta = theta - mu_new

        # sigma = delta.T.dot(np.diag(weights)).dot(delta) / Z
        sigma_new = delta.T.dot(np.diag(weights)).dot(delta) / Z

        mu_new_tmp = np.zeros_like(self._mu)
        mu_new_tmp[indices] = mu_new
        mu_new = mu_new_tmp

        sigma_new_tmp = sigma_old
        sigma_new_tmp[indices,indices] = np.diag(sigma_new)
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i is not j:
                    sigma_new_tmp[idx_i,idx_j] = sigma_new[i,j]
        sigma_new = sigma_new_tmp

        # store old mu and sigma
        mu = mu_old
        sigma = sigma_old

        # set new parameters
        self._mu = self._u.T @ mu_old - self._mu 
        sigma_new = self._u @ sigma_new @ self._vh
        self._u, self._s, self._vh = np.linalg.svd(sigma_new)
        self._mu = mu_new

        np.linalg.cholesky(sigma_new)


    def con_wmle_mi(self, theta, weights, eps, kappa, indices):
        # not really mi
        self._top_k = indices

        n_dims = len(self._mu)
        mu = self._mu
        sigma = self._u @ np.diag(self._s) @ self._vh

        eta_omg_opt_start = np.array([1., 1.])
        res = minimize(GaussianCholeskyDistribution._lagrangian_eta_omg, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa))
        
        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianCholeskyDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta_opt, omg_opt, kappa)

        self._u, self._s, self._vh = np.linalg.svd(sigma_new)
        self._mu = mu_new

        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)
        kl = GaussianCholeskyDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims)
        entropy = GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma, n_dims)

        return kl, entropy, self._mu

    def con_wmle_full(self, theta, weights, eps, kappa, indices):
        
        self._top_k = indices

        n_dims = len(self._mu)

        mu_old = np.copy(self._mu)
        s_old = np.copy(self._s)

        # get sigma in pca space and initialze mu
        mu = np.zeros_like(self._mu)
        sigma = np.diag(self._s)

        eta_omg_opt_start = np.array([1., 1.])
        res = minimize(GaussianCholeskyDistribution._lagrangian_eta_omg, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa))
        
        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianCholeskyDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta_opt, omg_opt, kappa)
        
        # transform back to original space and update mu
        mu_new = self._mu + self._u @ mu_new
        sigma_new = self._u @ sigma_new @ self._vh

        # store old mu and sigma
        mu = mu_old
        sigma = np.copy(self._u @ np.diag(s_old) @ self._vh)

        # set new parameters
        self._u, self._s, self._vh = np.linalg.svd(sigma_new)
        self._mu = mu_new
        
        # compute kl and entropy in original space
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)
        kl = GaussianCholeskyDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims)
        entropy = GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma, n_dims)
        return kl, entropy, self._mu

    def con_wmle_mi_diag(self, theta, weights, eps, kappa, indices):
        
        self._top_k = indices

        n_dims = len(self._mu)

        mu_old_true = np.copy(self._mu)
        s_old_true = np.copy(self._s)

        # get mu and sigma in pca space
        # mu = self._u.T @ self._vh.T @ self._mu
        mu = np.zeros_like(self._mu)
        sigma = np.sqrt(self._s)
        mu_old = np.copy(mu)
        sigma_old = np.copy(sigma)

        n_dims = len(indices)
        mu = mu[indices]
        sigma =  sigma[indices]
        theta = theta[:, indices]

        eta_omg_opt_start = np.array([1., 1.])
        res = minimize(GaussianDiagonalDistribution._lagrangian_eta_omg, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa))
        
        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianDiagonalDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta_opt, omg_opt, kappa)

        # project mu and sigma back to original space
        mu_new_tmp = np.copy(mu_old)
        mu_new_tmp[indices] = mu_new
        mu_new = self._mu + self._u @ mu_new_tmp

        sigma_new_tmp = np.copy(sigma_old)
        sigma_new_tmp[indices] = sigma_new
        sigma_new = self._u @ np.diag(sigma_new_tmp**2) @ self._vh

        # store old mu and sigma
        mu = mu_old_true
        sigma = np.copy(self._u @ np.diag(s_old_true) @ self._vh)

        # set new parameters
        self._u, self._s, self._vh = np.linalg.svd(sigma_new)
        self._mu = mu_new
        
        # logdet_sigma, n_dims

        # compute kl and entropy in original space
        n_dims = len(self._mu)
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)
        kl = GaussianDiagonalDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims)
        entropy = GaussianDiagonalDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - GaussianDiagonalDistribution._closed_form_entropy(logdet_sigma, n_dims)
        return kl, entropy, self._mu

    def con_wmle_mi_full(self, theta, weights, eps, kappa, indices):
        
        self._top_k = indices

        mu = np.zeros_like(self._mu)
        mu_old = np.copy(self._mu)
        sigma = np.diag(self._s)
        sigma_old = np.copy(np.diag(self._s))

        mu = mu[indices]
        sigma = np.diag(np.diag(sigma_old)[indices])
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i is not j:
                    sigma[i,j] = sigma_old[idx_i,idx_j]
        
        n_dims = len(mu)
        theta = theta[:, indices]

        eta_omg_opt_start = np.array([1, 1])
        res = minimize(GaussianCholeskyDistribution._lagrangian_eta_omg, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa))
        
        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianCholeskyDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta_opt, omg_opt, kappa)

        
        mu_new_tmp = np.zeros_like(self._mu)
        mu_new_tmp[indices] = mu_new
        mu_new = mu_new_tmp

        sigma_new_tmp = sigma_old
        sigma_new_tmp[indices,indices] = np.diag(sigma_new)
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i is not j:
                    sigma_new_tmp[idx_i,idx_j] = sigma_new[i,j]
        sigma_new = sigma_new_tmp

        # store old mu and sigma
        mu = mu_old
        sigma = sigma_old

        # set new parameters
        self._mu = self._u.T @ mu_old - self._mu 
        sigma_new = self._u @ sigma_new @ self._vh
        self._u, self._s, self._vh = np.linalg.svd(sigma_new)
        self._mu = mu_new

        np.linalg.cholesky(sigma_new)
        
        
        # compute kl and entropy in original space
        n_dims = len(self._mu)
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)
        kl = GaussianCholeskyDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims)
        entropy = GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma, n_dims)
        print(kl, entropy)
        return kl, entropy, self._mu

    def entropy(self):
        sigma = self._u @ np.diag(self._s) @ self._vh
        std = np.sqrt(np.diag(sigma))
        return 0.5 * np.log(np.product(2*np.pi*np.e*std**2))

    @property
    def parameters_size(self):
        n_dims = len(self._mu)

        return 2 * n_dims

class GaussianCholeskyDistribution(Distribution):
    """
    Gaussian distribution with full covariance matrix. The parameters
    vector represents the mean and the Cholesky decomposition of the
    covariance matrix. This parametrization enforce the covariance matrix to be
    positive definite.

    """
    def __init__(self, mu, sigma):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            sigma (np.ndarray): initial covariance matrix of the distribution.

        """
        self._mu = mu
        self._chol_sigma = np.linalg.cholesky(sigma)

        self._add_save_attr(
            _mu='numpy',
            _chol_sigma='numpy'
        )

    def sample(self):
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        return np.random.multivariate_normal(self._mu, sigma)

    def log_pdf(self, theta):
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        return multivariate_normal.logpdf(theta, self._mu, sigma)

    def __call__(self, theta):
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        return multivariate_normal.pdf(theta, self._mu, sigma)

    def entropy(self):
        std = np.diag(self._chol_sigma)
        return 0.5 * np.log(np.product(2*np.pi*np.e*std**2))

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
            sigma = np.cov(theta, rowvar=False)
        else:
            sumD = np.sum(weights)
            sumD2 = np.sum(weights**2)
            Z = sumD - sumD2 / sumD

            self._mu = weights.dot(theta) / sumD

            delta = theta - self._mu

            sigma = delta.T.dot(np.diag(weights)).dot(delta) / Z

        self._chol_sigma = np.linalg.cholesky(sigma)

    def con_wmle(self, theta, weights, eps, kappa):
        n_dims = len(self._mu)
        mu =self._mu
        sigma = self._chol_sigma.dot(self._chol_sigma.T)
        
        eta_omg_opt_start = np.array([1., 1.])
        res = minimize(GaussianCholeskyDistribution._lagrangian_eta_omg, eta_omg_opt_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(weights, theta, mu, sigma, n_dims, eps, kappa))
        
        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_new, sigma_new = GaussianCholeskyDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta_opt, omg_opt, kappa)

        self._mu, self._chol_sigma = mu_new, np.linalg.cholesky(sigma_new)

        #TODO: remove
        from tqdm import tqdm
        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)
        kl = GaussianCholeskyDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims)
        entropy = GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma, n_dims)
        # tqdm.write(f'\nKL constraint: KL: {kl:2.6f} eps: {eps:2.6f}')

        return kl, entropy, self._mu

    def diff_log(self, theta):
        n_dims = len(self._mu)
        inv_chol = np.linalg.inv(self._chol_sigma)
        inv_sigma = inv_chol.T.dot(inv_chol)

        g = np.empty(self.parameters_size)

        delta = theta - self._mu
        g_mean = inv_sigma.dot(delta)

        delta_a = np.reshape(delta, (-1, 1))
        delta_b = np.reshape(delta, (1, -1))

        S = inv_chol.dot(delta_a).dot(delta_b).dot(inv_sigma)

        g_cov = S - np.diag(np.diag(inv_chol))

        g[:n_dims] = g_mean
        g[n_dims:] = g_cov.T[np.tril_indices(n_dims)]

        return g

    def get_parameters(self):
        rho = np.empty(self.parameters_size)
        n_dims = len(self._mu)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._chol_sigma[np.tril_indices(n_dims)]

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        self._mu = rho[:n_dims]
        self._chol_sigma[np.tril_indices(n_dims)] = rho[n_dims:]

    @property
    def parameters_size(self):
        n_dims = len(self._mu)

        return 2 * n_dims + (n_dims * n_dims - n_dims) // 2

    @staticmethod
    def closed_form_mu1_sigma_new(*args):
        weights, theta, mu, sigma, n_dims, eps, eta, omg, kappa = args
        weights_sum = np.sum(weights)

        mu_new = (weights @ theta + eta * mu) / (weights_sum + eta)
        
        sigmawa = (theta - mu_new).T @ np.diag(weights) @ (theta - mu_new)
        sigma_new = (sigmawa + eta * sigma + eta * (mu_new - mu)[:, np.newaxis] @ (mu_new - mu)[:, np.newaxis].T) / (weights_sum + eta - omg)
        
        return mu_new, sigma_new

    @staticmethod
    def _closed_form_KL_constraint_M_projection(*args):
        mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims = args
       
        return 0.5*(np.trace(sigma_new_inv@sigma) - n_dims + logdet_sigma_new - logdet_sigma + (mu_new - mu).T @ sigma_new_inv @ (mu_new - mu))
    
    @staticmethod
    def _closed_form_entropy(*args):
        logdet_sigma, n_dims = args
        c = n_dims * np.log(2*np.pi)

        return 0.5 * (logdet_sigma + c + n_dims)

    @staticmethod
    def _lagrangian_eta_omg(lag_array, *args):
        weights, theta, mu, sigma, n_dims, eps, kappa = args
        eta, omg = lag_array[0], lag_array[1]

        mu_new, sigma_new = GaussianCholeskyDistribution.closed_form_mu1_sigma_new(weights, theta, mu, sigma, n_dims, eps, eta, omg, kappa)
        
        sigma_inv = np.linalg.inv(sigma)
        sigma_new_inv = np.linalg.inv(sigma_new)

        (sign_sigma, logdet_sigma) = np.linalg.slogdet(sigma)
        (sign_sigma_new, logdet_sigma_new) = np.linalg.slogdet(sigma_new)

        c = n_dims * np.log(2*np.pi)

        sum1 = np.sum([w_i * (-0.5 * (theta_i - mu_new)[:,np.newaxis].T @ sigma_new_inv @ (theta_i -  mu_new)[:,np.newaxis] - 0.5 * logdet_sigma_new - 0.5 * c) for w_i, theta_i in zip(weights, theta)])
        
        sum2 = eta * (eps - GaussianCholeskyDistribution._closed_form_KL_constraint_M_projection(mu, mu_new, sigma, sigma_new, sigma_inv, sigma_new_inv, logdet_sigma, logdet_sigma_new, n_dims))
        
        sum3 = omg * (GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma_new, n_dims) - ( GaussianCholeskyDistribution._closed_form_entropy(logdet_sigma, n_dims) - kappa ) )

        return sum1 + sum2 + sum3