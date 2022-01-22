import os
import argparse
import joblib
import numpy as np

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles

from mushroom_rl.environments import ShipSteering

import torch
from nes_shroom.nes_shroom import CoreNES
from nes_shroom.modules import ProMPNES, LinearRegressorNES
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor


def experiment( n_tilings, \
                alg, eps, kappa, k, \
                population_size, n_rollout, optim_lr,
                sigma_init, distribution, \
                method, mi_type, bins, sample_type, gamma, mi_avg, \
                n_epochs, fit_per_epoch, ep_per_fit, \
                seed, results_dir, quiet):

    quiet = bool(quiet)
    mi_avg = bool(mi_avg)
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    mdp = ShipSteering()

    # Policy
    high = [150, 150, np.pi]
    low = [0, 0, -np.pi]
    # low = [0, 0, -np.pi, -0.26]
    # high = [150, 150, np.pi, 0.26]
    n_tiles = [5, 5, 6]
    # n_tiles = [5, 5, 6, 5]
    low = np.array(low, dtype=float)
    high = np.array(high, dtype=float)

    tilings = Tiles.generate(n_tilings=n_tilings, n_tiles=n_tiles, 
                            low=low, high=high)

    features = Features(tilings=tilings)

    init_params = locals()

    policy = LinearRegressorNES(features.size, mdp.info.action_space.shape[0], population_size=population_size, l_decay=1., l2_decay=0., sigma=sigma_init, n_rollout=n_rollout, features=features)

    print('action space', mdp.info.action_space.shape)
    print('action lo/hi', mdp.info.action_space.low, mdp.info.action_space.high)
    print('observation space', mdp.info.observation_space.shape, 'features', features.size)

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
        n_tilings = 3,
        
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
        quiet = 1 # True
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n-tilings', type=int)

    parser.add_argument('--alg', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--k', type=int)

    parser.add_argument('--sigma-init', type=float)
    parser.add_argument('--distribution', type=str)

    parser.add_argument('--optim-lr', type=float)
    parser.add_argument('--population-size', type=int)
    parser.add_argument('--n-rollout', type=int)

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

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
