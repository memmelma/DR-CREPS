import numpy as np
from .policy import ParametricPolicy


class ProMPPolicy(ParametricPolicy):
    """
    Simple parametric policy representing a deterministic policy. As
    deterministic policies are degenerate probability functions where all
    the probability mass is on the deterministic action,they are not
    differentiable, even if the mean value approximator is differentiable.

    """
    def __init__(self, weights=None, n_basis=20, basis_width=0.005, c=None, maxSteps=1000, output=1, time_scale=1):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the action to select
                in each state.

        """

        self._step = 0
        self.trajectory = None

        self.dim = output[0]
        self.n_basis = n_basis

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.zeros(self.dim*self.n_basis)

        self.time_scale = time_scale
        self.basis_width = basis_width
        if c is not None:
            assert self.nBasis==len(c), "num of basis is not consistent with num of c"
            self.c = c
        else:
            self.c = np.linspace(-2 * self.basis_width, 1+2*self.basis_width, self.n_basis)
        self._maxSteps = maxSteps
        self.phi = lambda z: self.norm_gaussian_basis(z)

    def norm_gaussian_basis(self, z):
        features = np.exp(-0.5 * np.square(z - self.c[:, None]) / self.basis_width)
        return features / np.sum(features, axis=0)

    # def __call__(self, state, action):
    #     policy_action = self._approximator.predict(state, **self._predict_params)

    #     return 1. if np.array_equal(action, policy_action) else 0.

    def draw_action(self, state):
        assert self._step < self._maxSteps, 'step > maxSteps'
        return self.trajectory[:,self._step]

    def set_weights(self, weights):
        self.weights = weights
        self.sample_trajectory()

    def get_weights(self):
        return self.weights

    def sample_trajectory(self):
        weights = self.weights.reshape((self.dim, self.n_basis))
        maxStep = self._maxSteps*self.time_scale
        self.phase = np.linspace(0, 1, maxStep)
        assert np.shape(weights)==(self.dim, self.n_basis), "The size of weights should be [n_dim, n_basis]"
        self.trajectory = np.einsum('ij, jk->ik', weights, self.phi(self.phase))

    @property
    def weights_size(self):
        return self.weights.size
