import os
import argparse

import numpy as np
from numpy.random import default_rng

from tqdm import tqdm

from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.distributions import GaussianCholeskyDistribution, GaussianDistribution #, GaussianDiagonalDistribution
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.optimizers import AdaptiveOptimizer
# from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS
from reps_debug import REPS
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

# from constrained_REPS import constrained_REPS
# from more import MORE
from reps_mi import REPS_MI
from reps_mi_con import REPS_MI_CON
from constrained_REPS import REPS_CON

from gaussian_diag_custom import GaussianDiagonalDistribution

import matplotlib.pyplot as plt

import joblib
from mushroom_rl.environments import Environment
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

def experiment(alg, lqr_dim, eps, k, kappa, n_epochs, fit_per_epoch, ep_per_fit, sigma_init=1e-3, env_seed=42, n_ineff=-1, seed=42, results_dir='results', quiet=True):
    
    if n_ineff < 0:
        n_ineff = round(lqr_dim / 2)
    np.random.seed(seed)
    
    horizon = 50
    mdp = LQR.generate(dimensions=lqr_dim, horizon=horizon, episodic=False, max_pos=1., max_action=1., eps=0.5)

    if env_seed >= 0:
        rng = default_rng(seed=env_seed)
        ineff_params = rng.choice(lqr_dim, size=n_ineff, replace=False)

        for p in ineff_params:
            # mdp.A[p][p] = 1e-10
            mdp.Q[p][p] = 1e-10
            mdp.R[p][p] = 1e-10
            mdp.B[p][p] = 1e-10

        print('A', mdp.A)
        print('Q', mdp.Q)
        print('R', mdp.R)
        print('B', mdp.B)
        print('ineff_params', ineff_params)

    else:
        ineff_params = []
    
    # Calculate optimal control return
    gain_lqr = compute_lqr_feedback_gain(mdp)
    state = mdp.reset()
    optimal_reward = 0
    for i in range(horizon):
        action = - gain_lqr @ state
        state, reward, _, __ = mdp.step(action)
        optimal_reward += reward
    
    mdp.Q[mdp.Q == 1e-10] = 0
    mdp.R[mdp.R == 1e-10] = 0
    mdp.B[mdp.B == 1e-10] = 0
    
    # REMOVE
    # ineff_params = [2,3]

    init_params = locals()
    
    os.makedirs(results_dir, exist_ok=True)
    
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = DeterministicPolicy(mu=approximator)

    mu = np.zeros(policy.weights_size)

    # REMOVE
    # gain_lqr = compute_lqr_feedback_gain(mdp)
    # mu = gain_lqr.flatten()

    sigma = sigma_init * np.ones(policy.weights_size)
    
    distribution = GaussianDiagonalDistribution(mu, sigma)

    # algorithms
    if alg == 'REPS':
        alg = REPS
        params = {'eps': eps}

    elif alg == 'REPS_MI':
        alg = REPS_MI
        params = {'eps': eps, 'k': k}

    elif alg == 'REPS_MI_ORACLE':
        alg = REPS_MI
        oracle = []
        for i in range(lqr_dim):
            if i not in ineff_params:
                for j in range(lqr_dim):
                    oracle += [i*lqr_dim + j]
        print(oracle)
        params = {'eps': eps, 'k': k, 'oracle': oracle}

    # constrained
    elif alg == 'REPS_CON':
        alg = REPS_CON
        params = {'eps': eps, 'kappa': kappa}

    elif alg == 'REPS_MI_CON':
        alg = REPS_MI_CON
        params = {'eps': eps, 'k': k, 'kappa': kappa}

    elif alg == 'REPS_MI_CON_ORACLE':
        alg = REPS_MI_CON
        oracle = []
        for i in range(lqr_dim):
            if i not in ineff_params:
                for j in range(lqr_dim):
                    oracle += [i*lqr_dim + j]
        print(oracle)
        params = {'eps': eps, 'k': k, 'kappa': kappa, 'oracle': oracle}

    # covariance & sampling
    elif alg == 'REPS_MI_FIXED_LOW':
        alg = REPS_MI
        params = {'eps': eps, 'k': k}
        distribution.set_fixed_sample(eps=1e-10)

    elif alg == 'REPS_MI_FIXED_HIGH':
        alg = REPS_MI
        params = {'eps': eps, 'k': k}
        distribution.set_fixed_sample(eps=kappa)
        # distribution.set_fixed_sample(eps=1e-2)

    elif alg == 'REPS_MI_10':
        alg = REPS_MI
        params = {'eps': eps, 'k': k}
        distribution.set_percentage_sample(kappa=kappa)
        # distribution.set_percentage_sample(kappa=0.1)

    # Agent
    agent = alg(mdp.info, distribution, policy, **params)

    # Train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    
    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit, quiet=quiet)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet)
        # print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        returns_mean += [np.mean(J)]
        returns_std += [np.std(J)]
    

    gain_policy = policy.get_weights()

    mi_avg = None
    if alg.__name__ == 'REPS_MI' or alg.__name__ == 'REPS_MI_CON':
        mi_avg = agent.mis

    best_reward = np.array(returns_mean).max()

    mus = agent.mus
    kls = agent.kls

    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'agent': agent,
        'gain_lqr': gain_lqr,
        'gain_policy': gain_policy,
        'optimal_reward': optimal_reward,
        'best_reward': best_reward,
        'init_params': init_params,
        'alg': alg,
        'mi_avg': mi_avg,
        'mus': mus,
        'kls': kls,
        'ineff_params': ineff_params
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg.__name__}_{seed}'))
    
    filename = os.path.join(results_dir, f'log_{alg.__name__}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')


def default_params():
    defaults = dict(
        # alg = 'REPS_MI_CON_ORACLE', 
        alg = 'REPS_CON', 
        lqr_dim = 10, 
        eps = 1e-3,
        k = 8,
        kappa = 5,
        n_epochs = 50, 
        fit_per_epoch = 1, 
        ep_per_fit = 100,
        sigma_init=0.1,
        seed = 42,
        n_ineff = 7,
        env_seed = 0,
        results_dir = 'results',
        quiet = True
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--alg', type=str)
    parser.add_argument('--lqr-dim', type=int)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--k', type=int)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--fit-per-epoch', type=int)
    parser.add_argument('--ep-per-fit', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--env-seed', type=int)
    parser.add_argument('--n-ineff', type=int)
    parser.add_argument('--sigma-init', type=float)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--quiet', type=bool)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
