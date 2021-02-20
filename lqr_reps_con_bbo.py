import numpy as np
from tqdm import tqdm

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.distributions import GaussianCholeskyDistribution, GaussianDiagonalDistribution, GaussianDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

from constrained_REPS import constrained_REPS
from more import MORE

"""
This script aims to replicate the experiments on the LQR MDP using episode-based
policy search algorithms, also known as Black Box policy search algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, params, n_epochs, fit_per_epoch, ep_per_fit):
    np.random.seed()

    # MDP
    mdp = LQR.generate(dimensions=2, episodic=True, max_pos=1., max_action=1.)

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    # Cholesky
    sigma = 1e-3 * np.eye(policy.weights_size)
    distribution = GaussianCholeskyDistribution(mu, sigma)
    # Diag
    # std = 1e-3 * np.ones(policy.weights_size)
    # distribution = GaussianDiagonalDistribution(mu, std)
    # Gaussian w/ fixed cov
    # sigma = 1e-3 * np.eye(policy.weights_size)
    # distribution = GaussianDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_fit)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    for i in range(n_epochs):
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit)
        dataset_eval = core.evaluate(n_episodes=ep_per_fit)
        # print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))


if __name__ == '__main__':

    algs = [MORE]
    params = [{'eps': 0.5}] # beta is set in more.py according to method proposed in the MORE paper

    # algs = [constrained_REPS]
    # params = [{'eps': 0.5, 'kappa': 3.5}]
    
    # algs = [REPS]
    # params = [{'eps': 0.5}]
    

    for alg, params in zip(algs, params):
        print(alg.__name__)
        experiment(alg, params, n_epochs=10, fit_per_epoch=1, ep_per_fit=100)
