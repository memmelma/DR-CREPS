import numpy as np

from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

import autograd.numpy as np
from autograd import grad

import traceback

class REPS_con(BlackBoxOptimization):
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
            kappa (float): entropy constraint for the policy update. H(pi_t) - kappa <= H(pi_t+1)

        """
        self.eps = eps
        self.kappa = kappa

        self._add_save_attr(eps='primitive')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        
        # get distribution parameters mu and sigma
        n = len(self.distribution._mu)
        dist_params = self.distribution.get_parameters()
        mu_t = dist_params[:n]
        chol_sig_empty = np.zeros((n,n))
        chol_sig_empty[np.tril_indices(n)] = dist_params[n:]
        sig_t = chol_sig_empty.dot(chol_sig_empty.T)

        # REPS
        eta_start = np.ones(1)
        res = minimize(REPS_con._dual_function, eta_start,
                       jac=REPS_con._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self.eps, Jep, theta),
                       method=None)
        eta_opt = res.x.item()
        Jep -= np.max(Jep)
        W = np.exp(Jep / eta_opt)

        # optimize M-projection Langrangian
        eta_omg_start = np.ones(2)
        res = minimize(REPS_con._lagrangian_eta_omg, eta_omg_start,
                       jac=grad(REPS_con._lagrangian_eta_omg),
                       bounds=((0, np.inf),(0, np.inf)),
                       args=(W, theta, mu_t, sig_t, n, self.eps, self.kappa),
                       method=None)
        eta_opt, omg_opt  = res.x[0], res.x[1]
        # print('\neta_opt', eta_opt, 'omg_opt', omg_opt, 'optimizer success', res.success)

        # find closed form mu_t1 and sig_t1 using optimal eta and omg
        mu_t1, sig_t1 = REPS_con.closed_form_mu_t1_sig_t1(W, theta, mu_t, sig_t, n, self.eps, eta_opt, omg_opt, self.kappa)

        # check entropy constraint
        (sign_sig_t, logdet_sig_t) = np.linalg.slogdet(sig_t)
        (sign_sig_t1, logdet_sig_t1) = np.linalg.slogdet(sig_t1)
        H_t = REPS_con.closed_form_entropy(logdet_sig_t, n)
        H_t1 = REPS_con.closed_form_entropy(logdet_sig_t1, n)
        # print('H_t', H_t, 'H_t1', H_t1)
        if not H_t-self.kappa <= H_t1:
            print('entropy constraint violated', 'H_t', H_t, 'H_t1', H_t1, 'kappa', self.kappa)

        # check KL constraint
        sig_t_inv = np.linalg.inv(sig_t)
        sig_t1_inv = np.linalg.inv(sig_t1)
        kl = REPS_con.closed_form_KL(mu_t, mu_t1, sig_t, sig_t1, sig_t_inv, sig_t1_inv, logdet_sig_t, logdet_sig_t1, n)
        # print('KL', kl)
        if not kl <= self.eps:
            print('KL constraint violated', 'kl', kl, 'eps', self.eps)

        try:
            dist_params = np.concatenate((mu_t1.flatten(), np.linalg.cholesky(sig_t1)[np.tril_indices(n)].flatten()))
        except np.linalg.LinAlgError:
            traceback.print_exc()
            print('error in setting dist_params - sig_t1 not positive definite')
            print('mu_t1', mu_t1)
            print('sig_t1', sig_t1)
            print('eta_opt', eta_opt, 'omg_opt', omg_opt)
            exit(42)

        self.distribution.set_parameters(dist_params)

    @staticmethod
    def closed_form_mu_t1_sig_t1(*args):
        
        W, theta, mu_t, sig_t, n, eps, eta, omg, kappa = args
        W_sum = np.sum(W)

        mu_t1 = (W @ theta + eta * mu_t) / (W_sum + eta)
        # mu_t1 = (np.sum(W@theta) + eta * mu_t) / (W_sum + eta)
        # mu_t1 = (np.sum([w_i*theta_i for w_i, theta_i in zip(W, theta)]) + eta * mu_t) / (W_sum + eta)
        
        sig_wa = (theta - mu_t1).T @ np.diag(W) @ (theta - mu_t1) # moved .T to first part
        # sig_wa_ = np.sum([w_i*(theta_i - mu_t1)@(theta_i - mu_t1).T for w_i, theta_i in zip(W, theta)]) # (theta - mu_t1).T @ np.diag(W) @ (theta - mu_t1) # moved .T to first part
        
        sig_t1 = (sig_wa + eta * sig_t + eta * (mu_t1 - mu_t) @ (mu_t1 - mu_t).T) / (W_sum + eta - omg)

        return mu_t1, sig_t1

    @staticmethod
    # M-constraint KL(p_t||p_t1)
    def closed_form_KL(*args):
        mu_t, mu_t1, sig_t, sig_t1, sig_t_inv, sig_t1_inv, logdet_sig_t, logdet_sig_t1, n = args
        return 0.5*(np.trace(sig_t1_inv@sig_t) - n + logdet_sig_t1 - logdet_sig_t + (mu_t1 - mu_t).T @ sig_t1_inv @ (mu_t1 - mu_t))
    
    @staticmethod
    def closed_form_entropy(*args):
        logdet_sig, n = args
        c = REPS_con.get_c(n)
        return 0.5 * (logdet_sig + c + n)
    
    @staticmethod
    def get_c(args):
        n = args
        return n * np.log(2*np.pi)

    @staticmethod
    def _lagrangian_eta_omg(lag_array, *args):

        W, theta, mu_t, sig_t, n, eps, kappa = args
        eta, omg = lag_array[0], lag_array[1]

        mu_t1, sig_t1 = REPS_con.closed_form_mu_t1_sig_t1(W, theta, mu_t, sig_t, n, eps, eta, omg, kappa)

        # for 1 dim case -> expand sig_t1 from (1,) to (1,1)
        if len(sig_t1.shape) == 1:
            sig_t1 = sig_t1[:,np.newaxis]
        
        # inverse
        try:
            sig_t_inv = np.linalg.inv(sig_t)
            sig_t1_inv = np.linalg.inv(sig_t1)
        except np.linalg.LinAlgError:
            traceback.print_exc()
            print('error in lagrangian - cant inverse')
            print(sig_t1)
            exit(42)

        # log determinants
        (sign_sig_t, logdet_sig_t) = np.linalg.slogdet(sig_t)
        (sign_sig_t1, logdet_sig_t1) = np.linalg.slogdet(sig_t1)

        c = REPS_con.get_c(n)

        sum1 = np.sum([w_i * (-0.5*(theta_i - mu_t1)[:,np.newaxis].T @ sig_t1_inv @ (theta_i -  mu_t1)[:,np.newaxis] - 0.5 * logdet_sig_t1 - c/2) for w_i, theta_i in zip(W, theta)])
        
        # sum2 = eta * (eps - 0.5*(np.trace(sig_t1_inv@sig_t) - n + logdet_sig_t1 - logdet_sig_t + (mu_t1 - mu_t).T @ sig_t1_inv @ (mu_t1 - mu_t)))
        sum2 = eta * (eps - REPS_con.closed_form_KL(mu_t, mu_t1, sig_t, sig_t1, sig_t_inv, sig_t1_inv, logdet_sig_t, logdet_sig_t1, n))
        
        # beta = H_t - kappa
        # sum3 = omg * (H_t1 - beta)
        sum3 = omg * (REPS_con.closed_form_entropy(logdet_sig_t1, n) - REPS_con.closed_form_entropy(logdet_sig_t, n) + kappa)

        return sum1 + sum2 + sum3

    @staticmethod
    # alternative implementation
    def _lagrangian_eta_omg_2(lag_array, *args):
        W, theta, mu_t, sig_t, n, eps, kappa = args
        eta, omg = lag_array[0], lag_array[1]
        W_sum = np.sum(W)

        mu_t1, sig_t1 = REPS_con.closed_form_mu_t1_sig_t1(W, theta, mu_t, sig_t, n, eps, eta, omg, kappa)

        # for 1 dim case -> expand sig_t1 from (1,) to (1,1)
        if len(sig_t1.shape) == 1:
            sig_t1 = sig_t1[:,np.newaxis]
        
        try:
            sig_t1_inv = np.linalg.inv(sig_t1)
            sig_t_inv = np.linalg.inv(sig_t)
        except np.linalg.LinAlgError:
            traceback.print_exc()
            print('error in lagrangian - cant inverse')
            print(sig_t1)
            exit(42)

        sum1 = - 0.5 * np.sum([w_i * np.trace(sig_t1_inv @ (theta_i - mu_t1)[:,np.newaxis] @ (theta_i - mu_t1)[:,np.newaxis].T) for w_i, theta_i in zip(W,theta)])

        sum2 = - 0.5 * eta * np.trace(sig_t1_inv @ sig_t) 
        
        sum3 = - 0.5 * eta * np.trace(sig_t1_inv @ (mu_t1 - mu_t)[:,np.newaxis] @ (mu_t1 - mu_t)[:,np.newaxis].T)

        # log determinants
        (sign_sig_t, logdet_sig_t) = np.linalg.slogdet(sig_t)
        (sign_sig_t1, logdet_sig_t1) = np.linalg.slogdet(sig_t1)

        sum4 = 0.5 * (omg - eta - W_sum) * logdet_sig_t1
        sum5 = 0.5 * eta * (n + logdet_sig_t + 2 * eps)

        c = REPS_con.get_c(n)
        beta = REPS_con.closed_form_entropy(logdet_sig_t, n) - kappa
        
        sum6 = 0.5 * omg * (c + n - 2*beta) 
        sum7 = - 0.5 * W_sum * c
        
        return (sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7)

    @staticmethod
    def _dual_function(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J
        sum1 = np.mean(np.exp(r / eta))

        return eta * eps + eta * np.log(sum1) + max_J

    @staticmethod
    def _dual_function_diff(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J

        sum1 = np.mean(np.exp(r / eta))
        sum2 = np.mean(np.exp(r / eta) * r)

        gradient = eps + np.log(sum1) - sum2 / (eta * sum1)

        return np.array([gradient])
