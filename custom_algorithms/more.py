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
    def __init__(self, mdp_info, distribution, policy, eps, kappa, features=None):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            kappa (float): the upper bound for the entropy of the new policy.

        """
        # whether to use entropy constraint
        self.no_entropy = False

        self.eps = eps
        
        self.xi = -kappa
        self.kappa = 0 # -kappa

        if not isinstance(distribution, GaussianCholeskyDistribution):
            print('Only GaussianCholeskyDistribution is supported')
            exit()
    
        print('Init Entropy:', distribution.entropy())
   
        poly_basis_quadratic = PolynomialBasis().generate(2, policy.weights_size)
        self.phi_quadratic_ = Features(basis_list=poly_basis_quadratic)
        
        self.regressor_quadratic = Regressor(LinearApproximator,
                      input_shape=(len(poly_basis_quadratic),),
                      output_shape=(1,))

        poly_basis_linear = PolynomialBasis().generate(1, policy.weights_size)
        self.phi_linear_ = Features(basis_list=poly_basis_linear)

        self.regressor_linear = Regressor(LinearApproximator,
                      input_shape=(len(poly_basis_linear),),
                      output_shape=(1,))
        
        self._add_save_attr(eps='primitive')
        self._add_save_attr(kappa='primitive')

        self.kls = []
        self.entropys = []

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        
        if self.kappa != -1:
            gamma = 0.99
            entropy_policy_min = self.xi
            entropy_policy = self.distribution.entropy()
            self.kappa = gamma * (entropy_policy - entropy_policy_min) + entropy_policy_min
        
        # tqdm.write('Using different way to determine beta : distribution.entropy + kappa')
        self.kappa = self.distribution.entropy() + self.xi

        # get distribution parameters mu and sigma
        n = len(self.distribution._mu)
        dist_params = self.distribution.get_parameters()
        mu_t = dist_params[:n][:,np.newaxis]
        # cholesky sigma
        chol_sig_empty = np.zeros((n,n))
        chol_sig_empty[np.tril_indices(n)] = dist_params[n:]
        sig_t = chol_sig_empty.dot(chol_sig_empty.T)
        # # Gaussian sigma
        # sig_t = self.distribution._sigma

        R, r, r_0 = self.regression(theta, Jep, n)
        
        # # create polynomial features
        # features = self.phi_(theta)
        # Jep = ( Jep - np.mean(Jep, keepdims=True, axis=0) ) / np.std(Jep, keepdims=True, axis=0)
        
        # # fit linear regression
        # self.regressor.fit(features, Jep)

        # # get quadratic surrogate model from learned regression
        # R, r, r_0 = MORE._get_quadratic_model_from_regressor(self.regressor, n, theta, features)
        
        if self.no_entropy:
            self.kappa = -1
        
        # MORE lagrangian -> bounds from MORE page 3
        eta_omg_start = np.array([1, 1])
        res = minimize(MORE._dual_function, eta_omg_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(sig_t, mu_t, R, r, r_0, self.eps, self.kappa, n),
                       method=None)
                    #    method='SLSQP')

        eta_opt, omg_opt = res.x[0], res.x[1]
        if not res.success:
            tqdm.write(f'eta_opt {eta_opt} omg_opt {omg_opt} no optimizer success {res}')

        # # MLE update using closed form solution -> REPS Compendium page 17 equation 77

        if self.no_entropy:
            omg_opt = 0
        # calculate closed form solution
        mu_t1, sig_t1 = MORE._closed_form_mu_t1_sig_t1(sig_t, mu_t, R, r, eta_opt, omg_opt)

        # round to decimal for constraint checks
        dec = 4
        # check entropy constraint
        H_t1 = MORE._closed_form_entropy(sig_t1, n)
        if not np.round(H_t1,dec) >= np.round(self.kappa,dec) and not self.no_entropy:
            tqdm.write(f'entropy constraint violated kappa {self.kappa} >= H_t1 {H_t1}')

        # check KL constraint
        kl = MORE._closed_form_KL(mu_t1, mu_t, sig_t1, sig_t, n)
        if not np.round(kl,dec) <= np.round(self.eps,dec):
            tqdm.write(f'KL constraint violated KL {np.round(kl,4)[0][0]} eps {self.eps}')
        
        # update cholesky distribution
        dist_params = np.concatenate((mu_t1.flatten(), np.linalg.cholesky(sig_t1)[np.tril_indices(n)].flatten()))
        self.distribution.set_parameters(dist_params)

        # # update Gaussian distribution
        # self.distribution._mu = mu_t1.flatten()
        # self.distribution._sigma = sig_t1

        self.kls += [kl]
        # self.entropys += [H_t1-self.kappa]
        tqdm.write(f'policy change {np.round(H_t1-entropy_policy,4)} | kappa {self.xi} | using beta = distribution.entropy + kappa')
        tqdm.write(f'kl change {np.round(kl,4)[0][0]} | eps {self.eps}')
        self.entropys += [H_t1-entropy_policy]

    @staticmethod
    def _dual_function(lag_array, Q, b, R, r, r_0, eps, kappa, n):
        eta, omg = lag_array[0], lag_array[1]
        F, f = MORE._get_F_f(Q, b, R, r, eta)

        if kappa != -1:
            # original paper ( no **n in logdet & no r_0 ) added + r_0 instead of normalization
            slogdet_0 = np.linalg.slogdet( (2*np.pi) * Q )
            slogdet_1 = np.linalg.slogdet( (2*np.pi) * (eta + omg) * F )
            term1 = (f.T @ F @ f) - eta * (b.T @ np.linalg.inv(Q) @ b) - eta * slogdet_0[1] + (eta + omg) * slogdet_1[1] + r_0

            # REPS compendium
            # slogdet_0 = np.linalg.slogdet( (2*np.pi)**n * Q )
            # slogdet_1 = np.linalg.slogdet( (2*np.pi)**n * (eta + omg) * F )
            # term1 = (f.T @ F @ f) - eta * (b.T @ np.linalg.inv(Q) @ b) - eta * slogdet_0[1] + (eta + omg) * slogdet_1[1] + r_0
            
            return eta * eps - omg * kappa + 0.5 * term1[0]

        else:
            # REPS compendium: MORE w/o Entropy Constraint (or REPS with Quadratic Model)
            # slogdet_0 = np.linalg.slogdet( (2.*np.pi)**n * Q )
            # slogdet_1 = np.linalg.slogdet( (2.*np.pi)**n * eta * F )

            # adaptation ( no **n in logdet & no r_0 )
            slogdet_0 = np.linalg.slogdet( (2.*np.pi) * Q )
            slogdet_1 = np.linalg.slogdet( (2.*np.pi) * eta * F )

            term1 = (f.T @ F @ f) - eta * (b.T @ np.linalg.inv(Q) @ b) - eta * slogdet_0[1] + eta * slogdet_1[1] + r_0
            
            return eta * eps + 0.5 * term1[0]
    
    @staticmethod
    def _closed_form_mu_t1_sig_t1(Q, b, R, r, eta, omg):
        F, f = MORE._get_F_f(Q, b, R, r, eta)
        mu_t1 = F @ f
        sig_t1 = F * (eta + omg)
        return mu_t1, sig_t1

    @staticmethod
    def _get_F_f(Q, b, R, r, eta):
        Q_inv = np.linalg.inv(Q)
        F = np.linalg.inv(eta * Q_inv - 2. * R)
        f = eta * Q_inv @ b + r
        return F, f
    
    def regression(self, theta, Jep, n):

        # normalize outputs -> proved to be most effective
        Jep = ( Jep - np.mean(Jep, keepdims=True, axis=0) ) / np.std(Jep, keepdims=True, axis=0)
        # theta = ( theta - np.mean(theta, keepdims=True, axis=0) / np.std(theta, keepdims=True, axis=0))
        
        # create polynomial features
        features_quadratic = self.phi_quadratic_(theta)
        
        # normalize features, not bias
        # features_quadratic[:,1:] = ( features_quadratic[:,1:] - np.mean(features_quadratic, keepdims=True, axis=0)[:,1:] ) / np.std(features_quadratic, keepdims=True, axis=0)[:,1:]

        # fit linear regression
        self.regressor_quadratic.fit(features_quadratic, Jep)

        # get quadratic surrogate model from learned regression
        R, r, r_0 = MORE._get_quadratic_model_from_regressor(self.regressor_quadratic, n, theta, features_quadratic)
        
        # validate
        # theta_pred = theta[0][np.newaxis,:]
        # quadratic_pred_quad = theta_pred @ R @ theta_pred.T + theta_pred @ r + r_0
        # quadratic_regressor_pred = self.regressor_quadratic.predict(features_quadratic[0])[0]
        # tqdm.write(f'quadratic regression {quadratic_pred_quad.mean().round(4)} {quadratic_regressor_pred.mean().round(4)}')

        # force R to be negative definite
        w, v = np.linalg.eig(R)
        w[w >= 0.0] = -1e-12
        R = v @ np.diag(w) @ v.T
        R = 0.5 * (R + R.T)

        # create polynomial features
        
        features_linear = self.phi_linear_(theta)
        # normalize features, not bias
        # features_linear[:,1:] = ( features_linear[:,1:] - np.mean(features_linear, keepdims=True, axis=0)[:,1:] ) / np.std(features_linear, keepdims=True, axis=0)[:,1:]

        # refit linear regression w/o quadratic term
        # Jep - theta @ R @ theta.T = Jep w/o quadratic term/features
        aux = Jep - np.einsum('nk,kh,nh->n', theta, R, theta)
        self.regressor_linear.fit(features_linear, aux)
        
        beta = self.regressor_linear.get_weights()
        r_0 = beta[0]
        r = beta[1:][:,np.newaxis]

        # validate
        # theta_pred = theta[0][np.newaxis,:]
        # linear_pred_quad = theta_pred @ r + r_0 + (theta_pred @ R @ theta_pred.T)
        # linear_regressor_pred = self.regressor_linear.predict(features_linear[0])[0] + (theta_pred @ R @ theta_pred.T)
        # tqdm.write(f'linear regression {linear_pred_quad.mean().round(4)} {linear_regressor_pred.mean().round(4)}')

        return R, r, r_0

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
