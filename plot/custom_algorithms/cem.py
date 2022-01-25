import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.parameters import LinearParameter


class CEM(BlackBoxOptimization):
    """
    Cross Entropy Method.
    """
    def __init__(self, mdp_info, distribution, policy, eps, soft_update_factor=1., extra_expl_decay_iter=5, extra_expl_std_init=0., features=None):
        """
        Constructor.
        Args:
            eps ([int, Parameter]): # of samples to select as elites.

        [1] P.T. de Boer, D.P. Kroese, S. Mannor, R.Y. Rubinstein, "A Tutorial on the Cross-Entropy Method",
        Annals OR, 2005
        [2] I. Szita, A. Lörnicz, "Learning Tetris Using the NoisyCross-Entropy Method", Neural Computation, 2006

        https://github.com/famura/SimuRLacra/blob/master/Pyrado/pyrado/algorithms/episodic/cem.py
        
        """
        self._samples = to_parameter(int(eps))
        self._soft_update_factor = to_parameter(soft_update_factor)

        self._current_iteration = LinearParameter(0., threshold_value=extra_expl_decay_iter, n=extra_expl_decay_iter)
        self._extra_expl_decay_iter = extra_expl_decay_iter

        self._extra_expl_std_init = extra_expl_std_init
        self._add_save_attr(_samples='mushroom')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        
        # descending order of rewards
        idcs_dcs = np.argsort(-1*Jep)

        # select elites [1, p.12]
        theta_dcs = theta[idcs_dcs]
        theta_elites = theta_dcs[:self._samples()]

        # fit distribution
        self.distribution._mu = self._soft_update_factor() * np.mean(theta_elites, axis=0) + (1-self._soft_update_factor()) * self.distribution._mu
        self.distribution._std = np.std(theta_elites, axis=0) + \
            (1 - self._current_iteration() / self._extra_expl_decay_iter) * self._extra_expl_std_init**2 # [2, p.4]

        
        
