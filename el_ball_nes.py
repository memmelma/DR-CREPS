import os
import argparse
import joblib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from custom_env.ball_rolling_gym_env import BallRollingGym
from custom_policy.promp_policy import ProMPPolicy

from utils_el import init_distribution, init_algorithm, log_constraints

import torch
from nes_shroom.nes_shroom import CoreNES
from nes_shroom.modules import ProMPNES, LinearRegressorNES
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

def experiment( n_basis, horizon, \
                alg, eps, kappa, k, \
                population_size, n_rollout, optim_lr, \
                sigma_init, distribution, \
                method, mi_type, bins, sample_type, gamma, mi_avg, \
                n_epochs, fit_per_epoch, ep_per_fit, \
                seed, results_dir, quiet, save_render_path):
                
    quiet = bool(quiet)
    mi_avg = bool(mi_avg)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    mdp = BallRollingGym(horizon=horizon, gamma=0.99, observation_ids=[0,1,2,3], render=not quiet, save_render_path=save_render_path)

    policy = ProMPNES(mdp.info.observation_space.shape[0], mdp.info.action_space.shape[0], 
                        population_size=population_size, l_decay=1., l2_decay=0., sigma=sigma_init, n_rollout=n_rollout, 
                        features=None, n_basis=n_basis, basis_width=1e-3, maxSteps=horizon)

    print('action space', mdp.info.action_space.shape)
    print('action lo/hi', mdp.info.action_space.low, mdp.info.action_space.high)
    print('observation space', mdp.info.observation_space.shape)

    # prepro = StandardizationPreprocessor(mdp_info=mdp.info)
    # nes = CoreNES(policy, mdp, alg=alg, optimizer=torch.optim.Adam, optimizer_lr=optim_lr,
    #                 n_step=(n_epochs), prepro=prepro, seed=seed)
    nes = CoreNES(policy, mdp, alg=alg, optimizer=torch.optim.Adam, optimizer_lr=optim_lr,
                    n_step=(n_epochs), seed=seed)

    nes.train()

    init_params = nes.init_params

    dump_dict = dict({
        'returns_mean': nes.rewards_mean,
        'best_reward': nes.best_reward,
        'init_params': init_params,
        'alg': alg
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg}_{seed}'))
    
    filename = os.path.join(results_dir, f'log_{alg}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')
            file.write(f'{key}: {init_params[key]}\n')

def default_params():
    defaults = dict(
        # environment
        n_basis = 20,
        horizon = 750,

        # algorithm
        alg = 'NES',
        eps = 0,
        kappa = 0,
        k = 0,
        population_size = 6,
        n_rollout = 2,
        optim_lr = 0.02,

        # distribution
        sigma_init = 1e-0,
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
        n_epochs = 2, # 2,
        # n_epochs = 10,
        fit_per_epoch = 1, 
        # ep_per_fit = 100,
        ep_per_fit = 50, # 250,

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

    parser.add_argument('--optim-lr', type=float)
    parser.add_argument('--population-size', type=int)
    parser.add_argument('--n-rollout', type=int)

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
