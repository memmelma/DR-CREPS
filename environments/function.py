import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces


class Function(Environment):
    """
    Episodic functions to optimize. Largely adapted from: https://github.com/hanyas/reps/blob/master/reps/envs/episodic/benchmarks.py
    """
    def __init__(self, function, dim):
        """
        Constructor.
        """

        self._function = function
        self.dim = dim

        if self._function == 'himmelblau':
            print('Himmelblau, setting dim to 2')
            self.dim = 2

        observation_space = spaces.Box(low=-np.inf*np.ones(self.dim), high=np.inf*np.ones(self.dim))
        action_space = spaces.Box(low=-np.inf*np.ones(self.dim), high=np.inf*np.ones(self.dim))
        mdp_info = MDPInfo(observation_space, action_space, gamma=0, horizon=1)

        super().__init__(mdp_info)

    def reset(self, state):
        return 1e-1*np.ones(self.dim)

    def rosenbrock(self, x):
        return - np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 +
                        (1 - x[:-1]) ** 2.0, axis=-1)

    def himmelblau(self, x):
        a = x[0] * x[0] + x[1] - 11.0
        b = x[0] + x[1] * x[1] - 7.0
        return -1.0 * (a * a + b * b)

    def styblinski(self, x):
        return - 0.5 * np.sum(x**4.0 - 16.0 * x**2 + 5 * x, axis=-1)
    
    def rastrigin(self, x):
        return - (10.0 * self.dim + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1))


    def step(self, action):
        if self._function == 'rosenbrock':
            x = self.rosenbrock(action)
        elif self._function == 'himmelblau':
            x = self.himmelblau(action)
        elif self._function == 'styblinski':
            x = self.styblinski(action)
        elif self._function == 'rastrigin':
            x = self.rastrigin(action)

        return action, x, False, {}
