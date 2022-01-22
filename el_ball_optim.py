import os
import argparse
import joblib
import mushroom_rl
import numpy as np
from numpy.random import default_rng

from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.environments import LQR
from mushroom_rl.policy import DeterministicPolicy
from custom_policy.promp_policy import ProMPPolicy

from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

from utils_el import init_distribution, init_algorithm, log_constraints

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_env.ball_rolling_gym_env import BallRollingGym
from scipy.optimize import minimize


def eval(weights, core, mdp):
    # core, mdp = args[0], args[1]
    core.agent.distribution._mu = weights
    core.ctr += 1
    dataset_eval = core.evaluate(n_episodes=1, quiet=0)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)[0]
    
    if core.ctr % 50 == 0:
        print('sample', core.ctr, 'return', J)

    core.rewards += [J]
    return -J

def experiment( n_basis, horizon, \
                alg, eps, kappa, k, \
                sigma_init, distribution, \
                method, mi_type, bins, sample_type, gamma, mi_avg, \
                n_epochs, fit_per_epoch, ep_per_fit, \
                seed, results_dir, quiet, save_render_path):
    
    quiet = bool(quiet)
    mi_avg = bool(mi_avg)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    mdp = BallRollingGym(horizon=horizon, gamma=0.99, observation_ids=[0,1,2,3], render=not quiet, save_render_path=save_render_path)

    alg_name = alg

    policy = ProMPPolicy(n_basis=n_basis, basis_width=1e-3, maxSteps=horizon, output=mdp.info.action_space.shape)

    print('action space', mdp.info.action_space.shape)
    print('action lo/hi', mdp.info.action_space.low, mdp.info.action_space.high)
    print('observation space', mdp.info.observation_space.shape)
    print('parameters', policy.weights_size)

    # sample initial weigths like
    distribution = init_distribution(mu_init=0., sigma_init=sigma_init, size=policy.weights_size, sample_type=None, gamma=0., distribution_class='diag')
    weights_init = distribution.sample()

    distribution = init_distribution(mu_init=0., sigma_init=0., size=policy.weights_size, sample_type=None, gamma=0., distribution_class='diag')
    alg, params = init_algorithm(algorithm_class='REPS', params=init_params)
    agent = alg(mdp.info, distribution, policy, **params)
    core = Core(agent, mdp)
    core.rewards = []
    core.ctr = 0
    
    if alg_name == 'NM':
        res = minimize(eval, weights_init, args=(core, mdp), method='nelder-mead', options=dict({'maxiter': n_epochs*ep_per_fit}))
    elif alg_name =='BFGS':
        res = minimize(eval, weights_init, args=(core, mdp), method='L-BFGS-B', options=dict({'maxfun': n_epochs*ep_per_fit}))
    else:
        res = None

    best_weights = res.x
    print('J at end : ' + str(eval(best_weights, core, mdp))) # add [:-1] if final evaluation

    dump_dict = dict({
        'returns_mean': core.rewards[:-1],
        'returns_std': np.zeros_like(core.rewards[:-1]),
        'best_reward': max(core.rewards),
        'init_params': init_params,
        'alg': alg
    })
    
    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg_name}_{seed}'))
    
    filename = os.path.join(results_dir, f'log_{alg_name}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')


def default_params():
    defaults = dict(
        
        # environment
        n_basis = 20,
        horizon = 750,

        # algorithm
        alg = 'BFGS',
        eps = 0,
        kappa = 0,
        k = 0,

        # distribution
        sigma_init = 1e-0,
        distribution = 'diag',

        # MI related
        method = 'None',
        mi_type = 'None',
        bins = 0,
        sample_type = 'None',
        gamma = 0,
        mi_avg = 0, # False

        # training
        n_epochs = 10,
        fit_per_epoch = 1, 
        ep_per_fit = 500,

        # misc
        seed = 0,
        results_dir = 'results',
        quiet = 1, # True
        save_render_path = None
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n-basis', type=int)
    parser.add_argument('--horizon', type=int)

    parser.add_argument('--alg', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--k', type=int)

    parser.add_argument('--sigma-init', type=float)
    parser.add_argument('--distribution', type=str)

    parser.add_argument('--method', type=str)
    parser.add_argument('--mi-type', type=str)
    parser.add_argument('--bins', type=int)
    parser.add_argument('--sample-type', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--mi-avg', type=int)

    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--fit-per-epoch', type=int)
    parser.add_argument('--ep-per-fit', type=int)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--quiet', type=int)
    parser.add_argument('--save-render-path', type=int)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
