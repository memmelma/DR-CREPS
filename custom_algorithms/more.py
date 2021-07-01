from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

# from mushroom_rl.distributions import GaussianDiagonalDistribution, GaussianCholeskyDistribution, GaussianDistribution
# from mushroom_rl.distributions import Distribution
from custom_distributions.gaussian_custom import GaussianDiagonalDistribution, GaussianCholeskyDistribution, GaussianDistribution
from custom_distributions.distribution import Distribution

from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features.basis.polynomial import PolynomialBasis
from mushroom_rl.features import Features

# import autograd.numpy as np
# from autograd import grad
import numpy as np

import traceback
from tqdm import tqdm

class MORE(BlackBoxOptimization):
    """
    "Model-Based Relative Entropy Stochastic Search"
    Abdolmaleki, Abbas & Lioutikov, Rudolf & Peters, Jan & Lau, Nuno & Reis, Luís & Neumann, Gerhard.
    """
    def __init__(self, mdp_info, distribution, policy, eps, features=None):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            kappa (float): the upper bound for the entropy of the new policy.

        """
        self.eps = eps

        # set beta as specified in paper -> MORE page 4
        gamma = 0.99
        entropy_policy_min = -150 # 75
        entropy_policy = distribution.entropy()
        self.kappa = gamma * (entropy_policy - entropy_policy_min) + entropy_policy_min

        poly_basis = PolynomialBasis().generate(2, policy.weights_size)
        self.phi_ = Features(basis_list=poly_basis)
        
        self.regressor = Regressor(LinearApproximator,
                      input_shape=(len(poly_basis),),
                      output_shape=(1,))

        self._add_save_attr(eps='primitive')
        self._add_save_attr(kappa='primitive')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):

        # get distribution parameters mu and sigma
        n = len(self.distribution._mu)
        dist_params = self.distribution.get_parameters()
        mu_t = dist_params[:n][:,np.newaxis]

        # prepare sigma depending on distribution
        if isinstance(self.distribution, GaussianDiagonalDistribution):
            sig_t = np.diag(dist_params[n:]**2)
        elif isinstance(self.distribution, GaussianCholeskyDistribution):
            chol_sig_empty = np.zeros((n,n))
            chol_sig_empty[np.tril_indices(n)] = dist_params[n:]
            sig_t = chol_sig_empty.dot(chol_sig_empty.T)
        elif isinstance(self.distribution, GaussianDistribution):
            sig_t = self.distribution._sigma
        else:
            assert isinstance(self.distribution, Distribution), f'{self.distribution} is not a valid distribution!'
            print(self.distribution)

        # create polynomial features
        features = self.phi_(theta)
        # fit linear regression
        self.regressor.fit(features, Jep)

        # get quadratic surrogate model from learned regression
        R, r, r_0 = MORE._get_quadratic_model_from_regressor(self.regressor, n, theta, features)

        # MORE lagrangian -> bounds from MORE page 3
        eta_omg_start = np.array([1000, 1])
        res = minimize(MORE._dual_function, eta_omg_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(sig_t, mu_t, R, r, r_0, self.eps, self.kappa, n),
                       method='SLSQP')
        eta_opt, omg_opt = res.x[0], res.x[1]
        if not res.success:
            tqdm.write(f'eta_opt {eta_opt} omg_opt {omg_opt} no optimizer success {res}')

        # # MLE update using closed form solution -> REPS Compendium page 17 equation 77

        # calculate closed form solution
        mu_t1, sig_t1 = MORE._closed_form_mu_t1_sig_t1(sig_t, mu_t, R, r, eta_opt, omg_opt)
        
        tqdm.write(str(f'R{R}'))

        # round to decimal
        dec = 2
        # check entropy constraint
        H_t1 = MORE._closed_form_entropy(sig_t1, n)
        if not np.round(H_t1,dec) >= np.round(self.kappa,dec):
            tqdm.write(f'entropy constraint violated kappa {self.kappa} >= H_t1 {H_t1}')

        # check KL constraint
        kl = MORE._closed_form_KL(mu_t1, mu_t, sig_t1, sig_t, n)
        if not np.round(kl,dec) <= np.round(self.eps,dec):
            tqdm.write(f'KL constraint violated KL {kl} eps {self.eps}')

        # prepare parameters depending on distribution
        if isinstance(self.distribution, GaussianDiagonalDistribution):
            tqdm.write(f'sig_t1 {np.diag(sig_t1)}')
            dist_params = np.concatenate((mu_t1.flatten(), np.sqrt(np.diag(sig_t1)).flatten()))
        elif isinstance(self.distribution, GaussianCholeskyDistribution):
            dist_params = np.concatenate((mu_t1.flatten(), np.linalg.cholesky(sig_t1)[np.tril_indices(n)].flatten()))
        elif isinstance(self.distribution, GaussianDistribution):
            dist_params = mu_t1.flatten()
        
        # update distribution
        self.distribution.set_parameters(dist_params)

    @staticmethod
    def _dual_function(lag_array, Q, b, R, r, r_0, eps, kappa, n):
        eta, omg = lag_array[0], lag_array[1]
        F, f = MORE._get_F_f(Q, b, R, r, eta)
        
        # REPS compendium
        term1 = (f.T @ F @ f) - eta * (b.T @ np.linalg.inv(Q) @ b) - eta * (np.linalg.slogdet(((2*np.pi)**n)*Q)[1]) + (eta + omg) * (np.linalg.slogdet(((2*np.pi)**n)*(eta + omg) * F)[1]) + r_0
        # original paper: **n missing, performs better
        # term1 = (f.T @ F @ f) - eta * (b.T @ np.linalg.inv(Q) @ b) - eta * (np.linalg.slogdet(((2*np.pi))*Q)[1]) + (eta + omg) * (np.linalg.slogdet(((2*np.pi))*(eta + omg) * F)[1])

        return eta * eps - omg * kappa + 0.5 * term1[0]

    @staticmethod
    def _closed_form_mu_t1_sig_t1(Q, b, R, r, eta, omg):
        F, f = MORE._get_F_f(Q, b, R, r, eta)
        mu_t1 = F @ f
        sig_t1 = F * (eta + omg)
        return mu_t1, sig_t1

    @staticmethod
    def _get_F_f(Q, b, R, r, eta):
        F = np.linalg.inv(eta * np.linalg.inv(Q) - 2 * R)
        f = eta * np.linalg.inv(Q) @ b + r
        return F, f
    
    @staticmethod
    def _get_quadratic_model_from_regressor(regressor, n, theta, features):
        
        # get parameter vector beta
        beta = regressor.get_weights()

        # reconstruct components for quadratic surrogate model from beta
        r_0 = beta[0]
        r = beta[1:n+1][:,np.newaxis]

        R = np.zeros((n,n))
        beta_ctr = 1 + n
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    R[i][j] = beta[beta_ctr]
                else:
                    R[i][j] = beta[beta_ctr]/2
                    R[j][i] = beta[beta_ctr]/2
                beta_ctr += 1

        # verification that components got reconstructed correctly
        # theta_pred = np.atleast_2d(theta[0])
        # quadratic_pred = theta_pred @ R @ theta_pred.T + theta_pred @ r + r_0
        # regressor_pred = regressor.predict(features)[0]
        # tqdm.write(f'equal? {round(quadratic_pred[0][0],10) == round(regressor_pred[0],10)} quadratic model {quadratic_pred[0][0]} regressor {regressor_pred[0]}')

        return R, r, r_0

    # calculate entropy of policy
    @staticmethod
    def _closed_form_entropy(sig, n):
        '''
        Taken from page 11 of REPS compendium
        '''
        _, logdet_sig= np.linalg.slogdet(sig)
        c = n * np.log(2*np.pi)
        return 0.5 * (logdet_sig + c + n)
    
    @staticmethod
    def _closed_form_KL(mu_t, mu_t1, sig_t, sig_t1, n):
        _, logdet_sig_t = np.linalg.slogdet(sig_t)
        _, logdet_sig_t1 = np.linalg.slogdet(sig_t1)
        sig_t1_inv = np.linalg.inv(sig_t1)
        return 0.5*(np.trace(sig_t1_inv@sig_t) - n + logdet_sig_t1 - logdet_sig_t + (mu_t1 - mu_t).T @ sig_t1_inv @ (mu_t1 - mu_t))

    # manual polynomial regression (not in use)
    @staticmethod
    def get_phi(theta):
        # bias term
        phi = np.ones(theta.shape[0])[:, np.newaxis]
        # linear terms
        phi = np.concatenate((phi, theta), axis=1)
        # quadratic terms
        for i in range(theta.shape[1]):
            for j in range(i, theta.shape[1]):
                phi = np.concatenate((phi,(theta[:,i]*theta[:,j])[:, np.newaxis]), axis=1)
        
        return phi
    
    @staticmethod
    def get_prediction(theta, y):
        beta = MORE.get_linear_regression_beta(theta, y)
        phi = MORE.get_phi(theta)
        y_hat = phi @ beta
        return y_hat
    
    @staticmethod
    def get_linear_regression_beta(theta, y):
        print(y.shape)
        phi = MORE.get_phi(theta)
        beta = np.linalg.inv(phi.T @ phi) @ phi.T @ y
        print(beta.shape)
        return beta[:, np.newaxis]
