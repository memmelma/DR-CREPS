import os
import argparse

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
from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

# from constrained_REPS import constrained_REPS
# from more import MORE
# from reps_mi import REPS_MI

import matplotlib.pyplot as plt

import joblib
from mushroom_rl.environments import Environment
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

def experiment(alg, lqr_dim, params, n_epochs, fit_per_epoch, ep_per_fit, seed=42, results_dir='results', quiet=True):

    if alg == 'REPS':
        alg = REPS
    mdp = LQR.generate(dimensions=lqr_dim, episodic=True)

    init_params = locals()
    
    os.makedirs(results_dir, exist_ok=True)
    
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)
    sigma = 1e-3 * np.ones(policy.weights_size)
    distribution = GaussianDiagonalDistribution(mu, sigma)

    # Agent
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp)
    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    # print('J at start : ' + str(np.mean(J)))
    
    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit, quiet=quiet)
        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
        # print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        # print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        returns_mean += [np.mean(J)]
        returns_std += [np.std(J)]
    
    returns_mean = np.array(returns_mean)
    returns_std = np.array(returns_std)
    
    gain_lqr = compute_lqr_feedback_gain(mdp)
    gain_policy = policy.get_weights()

    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'agent': agent,
        'gain_lqr': gain_lqr,
        'gain_policy': gain_policy,
        'init_params': init_params
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg.__name__}_{seed}'))

    filename = os.path.join(results_dir, f'log_{alg.__name__}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')


def default_params():
    defaults = dict(
        alg = 'REPS', 
        lqr_dim = 3, 
        params = {'eps': 0.1}, 
        n_epochs = 100, 
        fit_per_epoch = 100, 
        ep_per_fit = 1, 
        seed = 42, 
        results_dir = 'results', 
        quiet = True
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    

    # arg_test = parser.add_argument_group('Test')
    # arg_test.add_argument("--a", type=int)
    # arg_test.add_argument("--b-c", type=int)
    # arg_test.add_argument("--boolean", action='store_true')
    # arg_test.add_argument('--default', type=str)

    # arg_default = parser.add_argument_group('Default')
    # arg_default.add_argument('--seed', type=int)
    # arg_default.add_argument('--results-dir', type=str)

    parser.add_argument('--alg', type=str)
    parser.add_argument('--mdp', type=int)
    parser.add_argument('--params', type=dict)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--fit_per_epoch', type=int)
    parser.add_argument('--ep_per_fit', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--quiet', type=bool)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
