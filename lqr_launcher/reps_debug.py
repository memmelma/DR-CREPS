import numpy as np

from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter


class REPS(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.
    """
    def __init__(self, mdp_info, distribution, policy, eps, features=None):
        """
        Constructor.
        Args:
            eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
        """
        self._eps = to_parameter(eps)

        self.mus = []
        self.kls = []

        self._add_save_attr(_eps='mushroom')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        eta_start = np.ones(1)

        res = minimize(REPS._dual_function, eta_start,
                       jac=REPS._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self._eps(), Jep, theta),
                       method='SLSQP')

        # print('eps', self._eps())
        # if not res.success:
        #     print('res', res)

        eta_opt = res.x.item()

        Jep -= np.max(Jep)

        d = np.exp(Jep / eta_opt)

        mu = self.distribution._mu
        std = self.distribution._std

        self.distribution.mle(theta, d)

        mu_new = self.distribution._mu
        std_new = self.distribution._std

        KL_full = REPS._closed_form_KL(mu, mu_new, np.diag(std), np.diag(std_new), len(mu))
        KL_full_M = REPS._closed_form_KL_M(mu, mu_new, np.diag(std), np.diag(std_new), len(mu))

        # print('KL', KL_full, KL_full_M)
        self.kls += [KL_full]

        self.mus += [self.distribution._mu]

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
    
    @staticmethod
    def _closed_form_KL(mu_t, mu_t1, sig_t, sig_t1, n):
        sig_t_inv = np.linalg.inv(sig_t)
        logdet_sig_t1 = np.linalg.slogdet(sig_t1)[1]
        logdet_sig_t = np.linalg.slogdet(sig_t)[1]
       
        return 0.5*(np.trace(sig_t_inv@sig_t1) - n + logdet_sig_t - logdet_sig_t1 + (mu_t - mu_t1).T @ sig_t_inv @ (mu_t - mu_t1))

    @staticmethod
    def _closed_form_KL_M(mu_t, mu_t1, sig_t, sig_t1, n):
        sig_t1_inv = np.linalg.inv(sig_t1)
        logdet_sig_t = np.linalg.slogdet(sig_t)[1]
        logdet_sig_t1 = np.linalg.slogdet(sig_t1)[1]
        return 0.5*(np.trace(sig_t1_inv@sig_t) - n + logdet_sig_t1 - logdet_sig_t + (mu_t1 - mu_t).T @ sig_t1_inv @ (mu_t1 - mu_t))