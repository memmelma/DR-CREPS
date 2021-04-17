import numpy as np

from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

import autograd.numpy as np
from autograd import grad

import traceback

class REPS_CON(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm with Constrained Policy Update (M-Projection Constraint).
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, eps, kappa, features=None):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            kappa (float): parameter for the entropy constraint of the policy 
                update to avoid premature convergence through slowly decreasing 
                the entropy of the policy. 
                math:: H(\pi_t) - \kappa \leq H(\pi_{t+1})

        """
        self.eps = eps
        self.kappa = kappa

        self.mus = []
        self.kls = []

        self._add_save_attr(eps='primitive')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        
        # Cholesky
        # get distribution parameters mu and sigma
        # n = len(self.distribution._mu)
        # dist_params = self.distribution.get_parameters()
        # mu_t = dist_params[:n]
        # chol_sig_empty = np.zeros((n,n))
        # chol_sig_empty[np.tril_indices(n)] = dist_params[n:]
        # sig_t = np.atleast_2d(chol_sig_empty.dot(chol_sig_empty.T))

        # Diag
        n = len(self.distribution._mu)
        dist_params = self.distribution.get_parameters()
        mu_t = dist_params[:n]
        diag_sig_empty = np.zeros((n,n))
        diag_sig_empty[np.diag_indices(n)] = dist_params[n:]
        sig_t = np.atleast_2d(diag_sig_empty)

        # Gaussian w/ fixed cov
        # n = len(self.distribution._mu)
        # mu_t = self.distribution.get_parameters()
        # sig_t = np.atleast_2d(self.distribution._sigma)

        # REPS
        eta_start = np.ones(1)
        res = minimize(REPS_CON._reps_dual_function, eta_start,
                       jac=REPS_CON._reps_dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self.eps, Jep, theta),
                       method='SLSQP')
        eta_opt = res.x.item()

        Jep -= np.max(Jep)
        W = np.exp(Jep / eta_opt)

        # optimize M-projection Langrangian
        eta_omg_start = np.ones(2)
        res = minimize(REPS_CON._lagrangian_eta_omg, eta_omg_start,
                       jac=grad(REPS_CON._lagrangian_eta_omg),
                       bounds=((0, np.inf),(0, np.inf)),
                       args=(W, theta, mu_t, sig_t, n, self.eps, self.kappa),
                       method='SLSQP')
        eta_opt, omg_opt  = res.x[0], res.x[1]

        # find closed form mu_t1 and sig_t1 using optimal eta and omg
        mu_t1, sig_t1 = REPS_CON.closed_form_mu_t1_sig_t1(W, theta, mu_t, sig_t, n, self.eps, eta_opt, omg_opt, self.kappa)
        
        # check entropy constraint
        (sign_sig_t, logdet_sig_t) = np.linalg.slogdet(sig_t)
        (sign_sig_t1, logdet_sig_t1) = np.linalg.slogdet(sig_t1)
        H_t = REPS_CON._closed_form_entropy(logdet_sig_t, n)
        H_t1 = REPS_CON._closed_form_entropy(logdet_sig_t1, n)
        # print('H_t', H_t, 'H_t1', H_t1)
        if not H_t-self.kappa <= H_t1:
            print('entropy constraint violated', 'H_t', H_t, 'H_t1', H_t1, 'kappa', self.kappa)

        # check KL constraint
        sig_t_inv = np.linalg.inv(sig_t)
        sig_t1_inv = np.linalg.inv(sig_t1)
        kl = REPS_CON._closed_form_KL_constraint_M_projection(mu_t, mu_t1, sig_t, sig_t1, sig_t_inv, sig_t1_inv, logdet_sig_t, logdet_sig_t1, n)
        # print('KL', kl)
        if not kl <= self.eps:
            print('KL constraint violated', 'kl', kl, 'eps', self.eps)
        
        # Cholesky
        # dist_params = np.concatenate((mu_t1.flatten(), np.linalg.cholesky(sig_t1)[np.tril_indices(n)].flatten()))
        # Diag
        dist_params = np.concatenate((mu_t1.flatten(), np.diag(sig_t1).flatten()))
        # Gaussian w/ fixed cov
        # dist_params = mu_t1.flatten()
        self.distribution.set_parameters(dist_params)

        self.mus += [self.distribution._mu]
        self.kls += [kl]
        
    @staticmethod
    def closed_form_mu_t1_sig_t1(*args):
        
        W, theta, mu_t, sig_t, n, eps, eta, omg, kappa = args
        W_sum = np.sum(W)

        mu_t1 = (W @ theta + eta * mu_t) / (W_sum + eta)
        sig_wa = (theta - mu_t1).T @ np.diag(W) @ (theta - mu_t1)
        sig_t1 = (sig_wa + eta * sig_t + eta * (mu_t1 - mu_t) @ (mu_t1 - mu_t).T) / (W_sum + eta - omg)

        return mu_t1, sig_t1

    @staticmethod
    def _closed_form_KL_constraint_M_projection(*args):
        mu_t, mu_t1, sig_t, sig_t1, sig_t_inv, sig_t1_inv, logdet_sig_t, logdet_sig_t1, n = args
        return 0.5*(np.trace(sig_t1_inv@sig_t) - n + logdet_sig_t1 - logdet_sig_t + (mu_t1 - mu_t).T @ sig_t1_inv @ (mu_t1 - mu_t))
    
    @staticmethod
    def _closed_form_entropy(*args):
        logdet_sig, n = args
        c = REPS_CON._get_c(n)
        return 0.5 * (logdet_sig + c + n)
    
    @staticmethod
    def _get_c(args):
        n = args
        return n * np.log(2*np.pi)

    @staticmethod
    def _lagrangian_eta_omg(lag_array, *args):

        W, theta, mu_t, sig_t, n, eps, kappa = args
        eta, omg = lag_array[0], lag_array[1]

        mu_t1, sig_t1 = REPS_CON.closed_form_mu_t1_sig_t1(W, theta, mu_t, sig_t, n, eps, eta, omg, kappa)
        
        # inverse
        sig_t_inv = np.linalg.inv(sig_t)
        sig_t1_inv = np.linalg.inv(sig_t1)

        # log determinants
        (sign_sig_t, logdet_sig_t) = np.linalg.slogdet(sig_t)
        (sign_sig_t1, logdet_sig_t1) = np.linalg.slogdet(sig_t1)

        c = REPS_CON._get_c(n)

        sum1 = np.sum([w_i * (-0.5*(theta_i - mu_t1)[:,np.newaxis].T @ sig_t1_inv @ (theta_i -  mu_t1)[:,np.newaxis] - 0.5 * logdet_sig_t1 - c/2) for w_i, theta_i in zip(W, theta)])
        
        sum2 = eta * (eps - REPS_CON._closed_form_KL_constraint_M_projection(mu_t, mu_t1, sig_t, sig_t1, sig_t_inv, sig_t1_inv, logdet_sig_t, logdet_sig_t1, n))
        
        sum3 = omg * (REPS_CON._closed_form_entropy(logdet_sig_t1, n) - REPS_CON._closed_form_entropy(logdet_sig_t, n) + kappa)

        return sum1 + sum2 + sum3

    @staticmethod
    def _reps_dual_function(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J
        sum1 = np.mean(np.exp(r / eta))

        return eta * eps + eta * np.log(sum1) + max_J

    @staticmethod
    def _reps_dual_function_diff(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J

        sum1 = np.mean(np.exp(r / eta))
        sum2 = np.mean(np.exp(r / eta) * r)

        gradient = eps + np.log(sum1) - sum2 / (eta * sum1)

        return np.array([gradient])
