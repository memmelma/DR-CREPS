import torch
from torch import nn
import numpy as np

from utils import *

class ShroomAgent(nn.Module):
    def __init__(self, input_size, output_size, population_size=256, n_rollout: int = 2, sigma=1e-3, lr=0.002, l_decay=0.999, l2_decay=0.005, features=None, discrete=False):
        super().__init__()

        self.population_size = population_size
        self.sigma = sigma
        self.n_rollout = n_rollout
        
        self.discrete = discrete

        self.l_decay = l_decay
        self.l2_decay = l2_decay
        self.lr = lr

        if features is not None:
            self.phi = features
        else:
            self.phi = None
    
    def sample(self):
        mean = nn.utils.parameters_to_vector(self.parameters())
        normal = torch.distributions.normal.Normal(0, self.sigma)
        epsilon = normal.sample([int(self.population_size/2), mean.shape[0]])
        population_params = torch.cat((mean + epsilon, mean - epsilon))
        epsilons = torch.cat((epsilon, -epsilon))
        return population_params, epsilons

    def calculate_gradients(self, rewards, epsilons):
        mean = nn.utils.parameters_to_vector(self.parameters())
        ranked_rewards = torch.from_numpy(rank_transformation(rewards)).unsqueeze(0)
        grad = -(torch.mm(ranked_rewards, epsilons) / (len(rewards) * self.sigma))
        grad = (grad + mean * self.l2_decay).squeeze()  # L2 Decay
        return grad

    def draw_action(self, state):
        raise NotImplementedError

    def no_grad_update(self, rewards, population_params, optim):
        lr = optim.param_groups[0]["lr"]

        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        
        update_factor = self.lr / (self.population_size * self.sigma)
        weights = torch.nn.utils.parameters_to_vector(self.parameters())
        weights = weights + update_factor * np.dot(population_params.T, rewards).T
        torch.nn.utils.vector_to_parameters(weights, self.parameters())
        optim.param_groups[0]["lr"] = self.lr * self.l_decay # Decay
        print(f'reduced lr to {optim.param_groups[0]["lr"]}')

class LinearRegressorNES(ShroomAgent):
    def __init__(self, input_size, output_size, population_size=256, n_rollout=2, sigma=1e-3, l_decay=0.999, l2_decay=0.005, features=None, discrete=False):
        super().__init__(input_size, output_size, population_size=population_size, n_rollout=n_rollout, sigma=sigma, l_decay=l_decay, l2_decay=l2_decay, features=features, discrete=discrete)
        
        self.w = nn.Parameter(torch.zeros((input_size, output_size)))
        

    def draw_action(self, state):
        if self.phi is not None:
            obs_tensor = torch.from_numpy(self.phi(state)).float()
        else:
            obs_tensor = torch.from_numpy(state).float()

        action = obs_tensor@self.w

        if self.discrete:
            return torch.tensor([0]) if action < 0.5 else torch.tensor([1])
        else:
            return action

    def set_weights(self, weights):
        if weights == self.parameters():
            return
        torch.nn.utils.vector_to_parameters(weights, self.parameters())
        
class ProMPNES(ShroomAgent):
    def __init__(self, input_size, output_size, population_size=256, n_rollout=2, sigma=1e-3, l_decay=0.999, l2_decay=0.005, features=None, discrete=False, n_basis=30, basis_width=1e-3, c=None, maxSteps=1000, time_scale=1):
        super().__init__(input_size, output_size, population_size=population_size, n_rollout=n_rollout, sigma=sigma, l_decay=l_decay, l2_decay=l2_decay, features=features, discrete=discrete)
        
        self.n_basis = n_basis
        self.basis_width = basis_width
        self.c = c
        self.maxSteps = maxSteps
        self.output = output_size
        self.time_scale = 1
        
        self._step = 0
        self.trajectory = None

        self.dim = output_size
        self.weights = nn.Parameter(torch.zeros(self.dim*self.n_basis))

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

    def sample_trajectory(self):
        weights = self.weights.reshape((self.dim, self.n_basis))
        maxStep = self._maxSteps*self.time_scale
        self.phase = np.linspace(0, 1, maxStep)
        assert np.shape(weights)==(self.dim, self.n_basis), "The size of weights should be [n_dim, n_basis]"
        self.trajectory = np.einsum('ij, jk->ik', weights, self.phi(self.phase))

    def draw_action(self, state):
        assert self._step < self._maxSteps, 'step > maxSteps'
        return torch.tensor(self.trajectory[:,self._step])
    
    def set_weights(self, weights):
        torch.nn.utils.vector_to_parameters(weights, self.parameters())
        self.sample_trajectory()
