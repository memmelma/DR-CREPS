import numpy as np

from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

from mushroom_rl.utils.parameters import ExponentialParameter

from sklearn.feature_selection import mutual_info_regression

class REPS_MI(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, eps, k, lr=1, features=None):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.

        """
        self.eps = eps
        self.k = k

        self.mis = []

        self.mi_avg = np.zeros(len(distribution._mu))
        self.lr = lr
        self.alpha = ExponentialParameter(1, exp=0.5)

        self._add_save_attr(eps='primitive')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        
        mi = mutual_info_regression(theta, Jep)
        self.mi_avg = self.mi_avg + self.lr * self.alpha() * ( mi - self.mi_avg )
        self.mis += [self.mi_avg]
      
        # print('\nMI for theta Jep: ',mi)
        # print('\nMI_avg for theta Jep: ',self.mi_avg)
        
        # top_k_mi = self.mi_avg > (np.sum(self.mi_avg)/len(self.mi_avg))
        
        top_k_mi = self.mi_avg.argsort()[-self.k:][::-1]
        # top_k_mi = self.mi_avg.argsort()[-int(theta.shape[1] * 0.3):][::-1]
        theta_mi = theta[:,top_k_mi]
        eta_start = np.ones(1)
        # exit(42)
        res = minimize(REPS_MI._dual_function, eta_start,
                       jac=REPS_MI._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self.eps, Jep, theta_mi))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)

        d = np.exp(Jep / eta_opt)
        
        # get new mu, std using wmle
        weights = d
        sumD = np.sum(weights)
        sumD2 = np.sum(weights**2)
        Z = sumD - sumD2 / sumD
        mu_mi = weights.dot(theta[:,top_k_mi]) / sumD
        delta2 = (theta[:,top_k_mi] - mu_mi)**2
        std_mi = np.sqrt(weights.dot(delta2) / Z)

        # get old mu, std
        mu = self.distribution._mu
        std = self.distribution._std
        
        # update using new mu, std
        mu_new = mu.copy()
        std_new = std.copy()
        mu_new[top_k_mi] = mu_mi
        std_new[top_k_mi] = std_mi

        # KL_full = REPS_MI._closed_form_KL(mu, mu_new, np.diag(std), np.diag(std_new), len(mu))
        # KL_reduced = REPS_MI._closed_form_KL(mu[top_k_mi], mu_new[top_k_mi], np.diag(std[top_k_mi]), np.diag(std_new[top_k_mi]), len(top_k_mi))
        # print('Equal?', round(KL_full,6) == round(KL_reduced,6), '| KL_full', KL_full, '| KL_reduced', KL_reduced)

        rho = np.concatenate((mu_new,std_new))
        self.distribution.set_parameters(rho)


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
        sig_t1_inv = np.linalg.inv(sig_t1)
        logdet_sig_t = np.linalg.slogdet(sig_t)[1]
        logdet_sig_t1 = np.linalg.slogdet(sig_t1)[1]
        return 0.5*(np.trace(sig_t1_inv@sig_t) - n + logdet_sig_t1 - logdet_sig_t + (mu_t1 - mu_t).T @ sig_t1_inv @ (mu_t1 - mu_t))