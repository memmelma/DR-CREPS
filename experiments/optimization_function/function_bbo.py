import numpy as np
from tqdm import tqdm, trange

from mushroom_rl.algorithms.policy_search import RWR, PGPE, REPS, ConstrainedREPS
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core, Logger
# from mushroom_rl.distributions import GaussianCholeskyDistribution, GaussianDiagonalDistribution, GaussianDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

from custom_algorithms.more import MORE
from custom_algorithms.constrained_reps_mi_full import ConstrainedREPSMIFull

from distributions.gaussian import GaussianDiagonalDistribution, GaussianCholeskyDistribution, GaussianDistributionMI

from policy.pass_policy import PassPolicy
from environments.function import Function
"""
This script aims to replicate the experiments on the LQR MDP using episode-based
policy search algorithms, also known as Black Box policy search algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, params, distribution, n_epochs, fit_per_epoch, ep_per_fit, function, dim):
    np.random.seed(0)

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)
    
    mdp = Function(function, dim=dim) # styblinski, himmelblau, rosenbrock, rastrigin
    policy = PassPolicy(mu_shape=mdp.info.action_space.shape)

    mu = np.zeros(policy.weights_size)

    if distribution == GaussianDiagonalDistribution:
        sigma = np.sqrt(1e-1) * np.ones(policy.weights_size)
    else:
        sigma = 1e-1 * np.eye(policy.weights_size)

    distribution = distribution(mu, sigma)

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
        # print('entropy', distribution.entropy())


if __name__ == '__main__':

    algs = [MORE, REPS]
    params = [{'eps':.05, 'kappa':.05}, {'eps':0.05}]
    distributions = [GaussianCholeskyDistribution, GaussianCholeskyDistribution]

    for alg, params, distribution in zip(algs, params, distributions):
        # function = styblinski, himmelblau, rosenbrock, rastrigin
        experiment(alg, params, distribution, n_epochs=200, fit_per_epoch=1, ep_per_fit=100, function='himmelblau', dim=15)
