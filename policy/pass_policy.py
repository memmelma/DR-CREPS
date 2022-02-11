import numpy as np
from mushroom_rl.policy import ParametricPolicy

class PassPolicy(ParametricPolicy):
    """
    Policy that returns its weights as action. Used for function optimization. 

    """
    def __init__(self, weight_size):
        """
        Constructor.

        Args:
            weight_size (int): weight size 1D.

        """
        self.weights = np.zeros(weight_size)

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
