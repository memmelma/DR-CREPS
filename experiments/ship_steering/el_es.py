import os
import numpy as np
import torch

from mushroom_rl.environments import ShipSteering

from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles

from algorithms import CoreNES, LinearRegressorNES

from experiments.utils import save_results

def experiment(
    env, seed, env_seed, \
    lqr_dim, red_dim, \
    n_tilings, \
    n_basis, horizon, \
    alg, eps, kappa, k, \
    distribution, sigma_init, \
    C, mi_estimator, \
    sample_strat, lambd, \
    nn_policy, actor_lr, critic_lr, max_kl, optim_eps, \
    n_rollout, population_size, optim_lr, \
    n_epochs, fit_per_epoch, ep_per_fit, \
    results_dir, save_render_path, verbose
):
    
    # misc
    verbose = bool(verbose==0)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

   # MDP
    mdp = ShipSteering()

    # features
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

    # policy
    policy = LinearRegressorNES(features.size, mdp.info.action_space.shape[0],
                    population_size=population_size, l_decay=1., l2_decay=0., sigma=sigma_init, n_rollout=n_rollout, features=features)
    
    # train
    nes = CoreNES(policy, mdp, alg=alg, optimizer=torch.optim.Adam, optimizer_lr=optim_lr,
                    n_step=(n_epochs), seed=seed)

    nes.train()

    # convert to plot format
    init_params['ep_per_fit'] = population_size * n_rollout

    # logging
    dump_dict = dict({
        'returns_mean': nes.rewards_mean,
        'optimal_reward': np.ones_like(nes.best_reward)*-59,
        'best_reward': nes.best_reward,
        'init_params': init_params,
        'alg': alg
    })

    save_results(dump_dict, results_dir, alg, init_params, seed)
    