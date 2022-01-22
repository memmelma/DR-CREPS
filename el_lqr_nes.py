import os
import argparse
import joblib
import numpy as np
from numpy.random import default_rng

from mushroom_rl.environments import LQR
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

import torch
from nes_shroom.nes_shroom import CoreNES
from nes_shroom.modules import ProMPNES, LinearRegressorNES
from utils_el import init_distribution, init_algorithm, log_constraints

def experiment( lqr_dim, n_ineff, env_seed, nn_policy, \
                population_size, n_rollout, optim_lr,
                alg, eps, kappa, k, \
                sigma_init, distribution, \
                method, mi_type, bins, sample_type, gamma, mi_avg, \
                n_epochs, fit_per_epoch, ep_per_fit, \
                seed, results_dir, quiet):
    
    quiet = bool(quiet)
    mi_avg = bool(mi_avg)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # init LQR
    horizon = 50
    mdp = LQR.generate(dimensions=lqr_dim, horizon=horizon, max_pos=1., max_action=1.)

    # init reduced LQR
    if env_seed >= 0:

        rng = default_rng(seed=0)
        choice = np.arange(0,lqr_dim,1)
        rng.shuffle(choice)

        ineff_params = choice[:n_ineff]
        eff_params = choice[n_ineff:]

        oracle = []
        tmp = np.arange(0,lqr_dim**2,1)
        for p in eff_params:
            oracle += tmp[p*lqr_dim:(p+1)*lqr_dim].tolist()
        init_params['oracle'] = oracle
        
        for p in ineff_params:
            mdp.B[p][p] = 1e-20
            mdp.Q[p][p] = 1e-20

    # env_seed < 0 for standard behavior
    else:
        ineff_params = []
    
    # compute optimal control return
    gain_lqr = compute_lqr_feedback_gain(mdp, max_iterations=100000) # 50
    state = mdp.reset()
    optimal_reward = 0
    for i in range(horizon):
        action = - gain_lqr @ state
        state, reward, _, __ = mdp.step(action)
        optimal_reward += reward
    print('optimal control', optimal_reward)
    
    oracle = np.where(np.abs(gain_lqr.flatten()) > 1e-5)[0]
    print(gain_lqr)
    init_params['oracle'] = [oracle]
    print('oracle params', len(oracle))

    policy = LinearRegressorNES(mdp.info.observation_space.shape[0], mdp.info.action_space.shape[0], population_size=population_size, l_decay=1., l2_decay=0., sigma=sigma_init, n_rollout=n_rollout, features=None)
    
    print(policy.w)
    # policy.weights_size = len(policy.w.flatten())
    # distribution = init_distribution(mu_init=0., sigma_init=sigma_init, size=policy.weights_size, sample_type=None, gamma=0., distribution_class='diag')
    # weights_init = distribution.sample()
    # print(weights_init.shape)
    # policy.set_weights(weights_init)

    print('action space', mdp.info.action_space.shape)
    print('action lo/hi', mdp.info.action_space.low, mdp.info.action_space.high)
    print('observation space', mdp.info.observation_space.shape)

    nes = CoreNES(policy, mdp, alg=alg, optimizer=torch.optim.Adam, optimizer_lr=optim_lr,
                    n_step=(n_epochs), seed=seed)

    nes.train()

    init_params = nes.init_params

    dump_dict = dict({
        'returns_mean': nes.rewards_mean,
        'gain_lqr': gain_lqr,
        'optimal_reward': optimal_reward,
        'best_reward': nes.best_reward,
        'init_params': init_params,
        'alg': alg,
        'ineff_params': ineff_params
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg}_{seed}'))
    
    filename = os.path.join(results_dir, f'log_{alg}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')


def default_params():
    defaults = dict(
        
        # environment
        lqr_dim = 10,
        n_ineff = 7,
        env_seed = 0,

        # algorithm
        alg = 'NES',
        eps = 0,
        kappa = 0,
        k = 0,
        population_size = 256,
        n_rollout = 2,
        optim_lr = 0.02,

        # distribution
        sigma_init = 3e-1,
        # distribution = 'cholesky',
        distribution = 'diag',

        # MI related
        method = 'None',
        mi_type = 'None',
        bins = 0,
        sample_type = 'None',
        gamma = 0,
        mi_avg = 0,

        # training
        # n_epochs = 50,
        # fit_per_epoch = 1, 
        # ep_per_fit = 60,
        n_epochs = 10, # 2,
        # n_epochs = 10,
        fit_per_epoch = 1, 
        # ep_per_fit = 100,
        ep_per_fit = 50, # 250,

        # misc
        seed = 0,
        results_dir = 'results',
        quiet = 1, # True
        nn_policy = 0 # False
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lqr-dim', type=int)
    parser.add_argument('--n-ineff', type=int)
    parser.add_argument('--env-seed', type=int)

    parser.add_argument('--alg', type=str)
    parser.add_argument('--eps', type=float)

    parser.add_argument('--optim-lr', type=float)
    parser.add_argument('--population-size', type=int)
    parser.add_argument('--n-rollout', type=int)

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
    parser.add_argument('--nn-policy', type=int)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
