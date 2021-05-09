import numpy as np

from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter

from mushroom_rl.utils.parameters import ExponentialParameter

from sklearn.feature_selection import mutual_info_regression

class ConstrainedREPSMI(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm with constrained policy update.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, eps, kappa, k, oracle=None, features=None):
        """
        Constructor.

        Args:
            eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            kappa ([float, Parameter]): the maximum admissible value for the entropy decrease
                between the new distribution and the 
                previous one at each update step. 

        """
        self._eps = to_parameter(eps)
        self._kappa = to_parameter(kappa)
        self._k = to_parameter(k)

        self.mis = []
        self.mi_avg = np.zeros(len(distribution._mu))
        self.alpha = ExponentialParameter(1, exp=0.5)

        self._add_save_attr(_eps='mushroom')
        self._add_save_attr(_kappa='mushroom')

        self.mus = []
        self.kls = []

        self.oracle = oracle

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        
        # REPS
        eta_start = np.ones(1)

        res = minimize(ConstrainedREPSMI._dual_function, eta_start,
                       jac=ConstrainedREPSMI._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self._eps(), Jep, theta))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)

        d = np.exp(Jep / eta_opt)
        
        # MI
        mi = mutual_info_regression(theta, Jep, discrete_features=False, n_neighbors=3, random_state=42)

        self.mi_avg = self.mi_avg + self.alpha() * ( mi - self.mi_avg )
        self.mis += [self.mi_avg]
        
        top_k_mi = self.mi_avg.argsort()[-self._k():][::-1]
        
        if self.oracle != None:
            top_k_mi = self.oracle

        # Constrained Update
        kl, mu = self.distribution.con_wmle_mi(theta, d, self._eps(), self._kappa(), top_k_mi)
        
        self.mus += [mu]
        self.kls += [kl]

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
