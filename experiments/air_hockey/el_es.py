import os
import numpy as np
import torch

from mushroom_rl.environments.pybullet_envs.air_hockey import AirHockeyHit

from algorithms import CoreNES, ProMPNES

from experiments.utils import init_distribution, save_results

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
    mdp = AirHockeyHit(horizon=horizon, debug_gui=not verbose, table_boundary_terminate=True)
    if save_render_path is not None:
        mdp.activate_save_render(save_render_path)

    # policy
    policy = ProMPNES(mdp.info.observation_space.shape[0], mdp.info.action_space.shape[0], 
                        population_size=population_size, l_decay=1., l2_decay=0., sigma=sigma_init, n_rollout=n_rollout, 
                        features=None, n_basis=n_basis, basis_width=1e-3, maxSteps=horizon)
    
    # intialize same as policy search
    policy.weights_size = len(policy.weights.flatten())
    distribution = init_distribution(mu_init=0., sigma_init=sigma_init, size=policy.weights_size, sample_strat=None, lambd=0., distribution_class='diag')
    policy.weights = torch.nn.Parameter(torch.tensor(distribution.sample()))

    # train
    nes = CoreNES(policy, mdp, alg=alg, optimizer=torch.optim.Adam, optimizer_lr=optim_lr,
                    n_step=(n_epochs), seed=seed)

    nes.train()

    # convert to plot format
    init_params['ep_per_fit'] = population_size * n_rollout

    # logging
    dump_dict = dict({
        'returns_mean': nes.rewards_mean,
        'optimal_reward': np.ones_like(nes.best_reward)*150.,
        'best_reward': nes.best_reward,
        'init_params': init_params,
        'alg': alg
    })

    save_results(dump_dict, results_dir, alg, init_params, seed)
    