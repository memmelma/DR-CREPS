import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm, trange

from custom_algorithms.more import MORE
from mushroom_rl.algorithms.policy_search import RWR, PGPE, REPS, ConstrainedREPS

from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core, Logger
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

# from mushroom_rl.distributions import GaussianCholeskyDistribution, GaussianDiagonalDistribution, GaussianDistribution
from custom_distributions.gaussian_custom import GaussianDiagonalDistribution, GaussianCholeskyDistribution

from mushroom_rl.environments import Gym

tqdm.monitor_interval = 0

class Network(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super(Network, self).__init__()
        hidden_features = in_features[0] // 2
        layers = [nn.Linear(in_features[0], hidden_features)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_features, out_features=out_features[0])]
        layers += [nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self,x) -> torch.Tensor:
        return self.layers(x)

def experiment(alg, params, n_epochs, fit_per_epoch, ep_per_fit):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = Gym('HopperBulletEnv-v0', horizon=None, gamma=0.99)
    
    approximator = Regressor(TorchApproximator,
                             network = Network,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)
                             
    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)

    # sigma = 1e-3 * np.eye(policy.weights_size)
    # sigma = 1e-1 * np.eye(policy.weights_size)
    # distribution = GaussianCholeskyDistribution(mu, sigma)

    # sigma = 1e-3 * np.eye(policy.weights_size)
    # distribution = GaussianDistribution(mu, sigma)

    sigma = 3e-1 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_fit)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    # logger.epoch_info(0, J=np.mean(J), distribution_parameters=distribution.get_parameters())
    logger.epoch_info(0, J=np.mean(J))

    for i in trange(n_epochs, leave=False):
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit)
        dataset_eval = core.evaluate(n_episodes=ep_per_fit)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        # logger.epoch_info(i+1, J=np.mean(J), distribution_parameters=distribution.get_parameters())
        logger.epoch_info(i+1, J=np.mean(J))
        print('entropy', distribution.entropy())


if __name__ == '__main__':
    optimizer = AdaptiveOptimizer(eps=0.05)

    algs = [REPS, MORE, ConstrainedREPS]
    params = [{'eps':1.5}, {'eps':0.7, 'kappa': 250}, {'eps':0.5, 'kappa':5}]

    # algs = [REPS, RWR, PGPE, ConstrainedREPS]
    # params = [{'eps': 0.5}, {'beta': 0.7}, {'optimizer': optimizer}, {'eps':0.5, 'kappa':5}]

    for alg, params in zip(algs, params):
        experiment(alg, params, n_epochs=5, fit_per_epoch=1, ep_per_fit=100)
