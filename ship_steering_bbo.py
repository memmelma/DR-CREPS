import numpy as np

from mushroom_rl.algorithms.policy_search import REPS, RWR, PGPE
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import ShipSteering
from mushroom_rl.features.tiles import Tiles
from mushroom_rl.features.features import Features
# from mushroom_rl.distributions import GaussianDiagonalDistribution
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

from custom_algorithms.constrained_reps_mi import ConstrainedREPSMI
from custom_distributions.gaussian_custom import GaussianDiagonalDistribution

from tqdm import tqdm

from custom_algorithms.more import MORE

"""
This script aims to replicate the experiments on the Ship Steering MDP 
using policy gradient algorithms.
"""

tqdm.monitor_interval = 0


def experiment(alg, params, n_epochs, fit_per_epoch, ep_per_fit):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    mdp = ShipSteering()

    # Policy
    high = [150, 150, np.pi, np.pi / 12.]
    low = [0, 0, -np.pi, -np.pi / 12.]
    n_tiles = [5, 5, 6, 3]

    low = np.array(low, dtype=np.float)
    high = np.array(high, dtype=np.float)
    n_tilings = 1

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, low=low,
                             high=high)

    phi = Features(tilings=tilings)
    input_shape = (phi.size,)

    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape)

    print('action space', mdp.info.action_space.low, mdp.info.action_space.high)
    policy = DeterministicPolicy(approximator)
    print('policy.weights_size', policy.weights_size)

    mu = np.zeros(policy.weights_size)
    sigma = 1e-1 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, features=phi, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_fit)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    logger.epoch_info(0, J=np.mean(J))

    for i in range(n_epochs):
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit)
        dataset_eval = core.evaluate(n_episodes=ep_per_fit)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        logger.epoch_info(i+1, J=np.mean(J))


if __name__ == '__main__':

    algs_params = [
        # (MORE, {'eps': 2.0})
        # (ConstrainedREPSMI, {'eps': 1.0, 'kappa':5, 'k':400, 'bins':4, 'gamma':0.1}),
        (REPS, {'eps': 1.0}),
        # (RWR, {'beta': 0.7}),
        # (PGPE, {'optimizer': AdaptiveOptimizer(eps=1.5)}),
        ]

    for alg, params in algs_params:
        experiment(alg, params, n_epochs=25, fit_per_epoch=1, ep_per_fit=20)
        # experiment(alg, params, n_epochs=200, fit_per_epoch=1, ep_per_fit=25)