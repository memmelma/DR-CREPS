import os
import numpy as np
import torch

from environments.ball_rolling_gym_env import BallRollingGym

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
    mdp = BallRollingGym(horizon=horizon, gamma=0.99, observation_ids=[0,1,2,3], render=not verbose, save_render_path=save_render_path)

    # policy
    policy = LinearRegressorNES(mdp.info.observation_space.shape[0], mdp.info.action_space.shape[0],
                    population_size=population_size, l_decay=1., l2_decay=0., sigma=sigma_init, n_rollout=n_rollout, features=None)
    
    # train
    nes = CoreNES(policy, mdp, alg=alg, optimizer=torch.optim.Adam, optimizer_lr=optim_lr,
                    n_step=(n_epochs), seed=seed)

    nes.train()

    # convert to plot format
    init_params['ep_per_fit'] = population_size * n_rollout

    # logging
    dump_dict = dict({
        'returns_mean': nes.rewards_mean,
        'best_reward': nes.best_reward,
        'init_params': init_params,
        'alg': alg
    })

    save_results(dump_dict, results_dir, alg, init_params, seed)
    