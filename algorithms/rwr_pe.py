import numpy as np

from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

from .utils import compute_corr

from distributions import GaussianDiagonalDistribution

class RWR_PE(BlackBoxOptimization):
    """
    Reward-Weighted Regression algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    Pearson-Correlation-Based Relevance Weighted PolicyOptimization (PRO)
    "Learning Trajectory Distributions for Assisted Teleoperation and Path Planning",
    Ewerton M., Arenz O., Maeda G., Koert D., Kolev Z.,  Takahashi M., Peters J.. 2019.

    """
    def __init__(self, mdp_info, distribution, policy, eps, lambd, k, C='PCC', mi_estimator=None, pro=True, features=None):
        """
        Constructor.

        Args:
            beta ([float, Parameter]): the temperature for the exponential reward
                transformation.

        """
        self._beta = to_parameter(eps)
        assert 0 <= lambd and lambd <= 1, f'lambd must be 0 \leq \lambda \leq 1 got {lambd}'
        self._lambda = to_parameter(lambd)
        self._k = to_parameter(k)

        self._mi_estimator = mi_estimator

        self.corr_list = []
        self.MI = np.zeros(len(distribution._mu))
        self.C = C
        
        self._add_save_attr(_beta='mushroom')
        self._add_save_attr(_lambda='mushroom')
        self._add_save_attr(_k='mushroom')

        super().__init__(mdp_info, distribution, policy, features)

        assert type(self.distribution) is GaussianDiagonalDistribution, f'Only GaussianDiagonalDistribution supports RWR with PE, got {type(self.distribution)}!'

        if pro:
            assert self.distribution._sample_strat == 'PRO', f"Please set 'sample_strat=PRO' if you wish to use PRO!"
            print('PRO selected, ignoring parameters: k, lambd, C, mi_estimator!')

    def _update(self, Jep, theta):
      
        self.distribution._lambda = self._lambda()

        Jep -= np.max(Jep)

        d = np.exp(self._beta() * Jep)

        self.top_k_corr, corr = compute_corr(theta, Jep, k=self._k(), C=self.C, estimator=self._mi_estimator)

        self.distribution.mle(theta, d, self.top_k_corr)
        self.distribution.update_importance(corr)#/ np.sum(corr)