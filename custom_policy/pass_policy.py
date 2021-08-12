import numpy as np
from .policy import ParametricPolicy


class PassPolicy(ParametricPolicy):
    """
    Simple parametric policy representing a deterministic policy. As
    deterministic policies are degenerate probability functions where all
    the probability mass is on the deterministic action,they are not
    differentiable, even if the mean value approximator is differentiable.

    """
    def __init__(self, mu_shape):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the action to select
                in each state.

        """
        self.weights = np.zeros(mu_shape)

    # def get_regressor(self):
    #     """
    #     Getter.

    #     Returns:
    #         The regressor that is used to map state to actions.

    #     """
    #     return self._approximator

    def __call__(self, state, action):
        policy_action = self.weights

        return 1. if np.array_equal(action, policy_action) else 0.

    def draw_action(self, state):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    @property
    def weights_size(self):
        return len(self.weights)
