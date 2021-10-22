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

from custom_algorithms.constrained_reps_mi_full import ConstrainedREPSMIFull

from custom_distributions.gaussian_custom import GaussianDiagonalDistribution, GaussianCholeskyDistribution, GaussianDistributionMI

# TODO
# from custom_algorithms.more import MORE
from custom_algorithms.more_shroom import MORE


"""
This script aims to replicate the experiments on the LQR MDP using episode-based
policy search algorithms, also known as Black Box policy search algorithms.

"""

tqdm.monitor_interval = 0


def experiment(alg, params, distribution, n_epochs, fit_per_epoch, ep_per_fit):
    np.random.seed(42)

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = LQR.generate(dimensions=10, horizon=50, episodic=False, max_pos=1., max_action=1.)

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    # sigma = 1e-3 * np.eye(policy.weights_size)
    # sigma = 1e-1 * np.eye(policy.weights_size)
    # distribution = GaussianCholeskyDistribution(mu, sigma)

    # sigma = 1e-3 * np.eye(policy.weights_size)
    # distribution = GaussianDistribution(mu, sigma)

    # sigma = 3e-1 * np.ones(policy.weights_size)
    # distribution = GaussianDiagonalDistribution(mu, sigma)
    
    if distribution == GaussianDiagonalDistribution:
        sigma = np.sqrt(1e-1) * np.ones(policy.weights_size)
    else:
        sigma = 1e-1 * np.eye(policy.weights_size)
    # distribution = GaussianDistributionMI(mu, sigma)

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
    optimizer = AdaptiveOptimizer(eps=0.05)

    algs = [MORE, REPS, REPS, ]
    params = [{'eps':2.5, 'mu':np.zeros(100), 'sigma': 1e-3*np.eye(100)}, {'eps':1.5}]
    distributions = [GaussianCholeskyDistribution, GaussianCholeskyDistribution]

    # algs = [ConstrainedREPSMIFull, REPS]
    # params = [{'eps':2.5, 'kappa':5., 'gamma':0.1, 'k':25, 'bins':4}, {'eps':2.5}]
    # distributions = [GaussianDistributionMI, GaussianCholeskyDistribution]

    # algs = [REPS, RWR, PGPE, ConstrainedREPS]
    # params = [{'eps': 0.5}, {'beta': 0.7}, {'optimizer': optimizer}, {'eps':0.5, 'kappa':5}]

    for alg, params, distribution in zip(algs, params, distributions):
        experiment(alg, params, distribution, n_epochs=5, fit_per_epoch=1, ep_per_fit=100)
