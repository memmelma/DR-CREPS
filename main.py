import numpy as np
from tqdm import tqdm

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.distributions import GaussianCholeskyDistribution, GaussianDiagonalDistribution, GaussianDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.algorithms.policy_search.black_box_optimization import REPS
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer

from mushroom_rl.features import Features
from mushroom_rl.features.basis.polynomial import PolynomialBasis
    
from custom_env.ball_rolling_gym_env import BallRollingGym
from custom_policy.promp_policy import ProMPPolicy
"""
This script aims to replicate the experiments on the LQR MDP using episode-based
policy search algorithms, also known as Black Box policy search algorithms.

"""

tqdm.monitor_interval = 0

def experiment(alg, params, n_epochs, fit_per_epoch, ep_per_fit):
    np.random.seed()

    horizon = 1000

    # MDP
    mdp = BallRollingGym(horizon=horizon, gamma=0.99, observation_ids=[0,1,2,3], render=False)

    # basis_features = PolynomialBasis().generate(2, mdp.info.observation_space.shape[0])
    # features = Features(basis_features)
    features=None
    
    # approximator = Regressor(LinearApproximator,
    #                          input_shape=(len(basis_features),),#mdp.info.observation_space.shape,
    #                          output_shape=mdp.info.action_space.shape)
    # policy = DeterministicPolicy(mu=approximator)
    
    policy = ProMPPolicy(n_basis=25, basis_width=0.001, maxSteps=horizon, output=mdp.info.action_space.shape)

    mu = np.zeros(policy.weights_size)
    # Cholesky
    # sigma = 1e-3 * np.eye(policy.weights_size)
    # distribution = GaussianCholeskyDistribution(mu, sigma)
    # Diag
    std = 1e-3 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, std)
    # Gaussian w/ fixed cov
    # sigma = 1e-3 * np.eye(policy.weights_size)
    # distribution = GaussianDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, features=features, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_fit)
    print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    for i in range(n_epochs):
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit)
        dataset_eval = core.evaluate(n_episodes=ep_per_fit)
        print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))


if __name__ == '__main__':

    # algs = [constrained_REPS]
    # params = [{'eps': 0.5, 'kappa': 3.5}]
    
    algs = [REPS]
    params = [{'eps': 0.5}]
    

    for alg, params in zip(algs, params):
        print(alg.__name__)
        experiment(alg, params, n_epochs=10, fit_per_epoch=1, ep_per_fit=25) 