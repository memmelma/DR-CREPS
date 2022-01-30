import os
import numpy as np
import torch

from algorithms import CoreNES, LinearRegressorNES

from environments import generate_red_LQR
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
    mdp, ineff_params, gain_lqr, optimal_reward = generate_red_LQR(lqr_dim, red_dim)    

    # policy
    policy = LinearRegressorNES(mdp.info.observation_space.shape[0], mdp.info.action_space.shape[0],
                    population_size=population_size, l_decay=1., l2_decay=0., sigma=sigma_init, n_rollout=n_rollout, features=None)
    
    # intialize same as policy search
    policy.weights_size = len(policy.weights.flatten())
    distribution = init_distribution(mu_init=0., sigma_init=sigma_init, size=policy.weights_size, sample_strat=None, lambd=0., distribution_class='diag')
    weights_init = torch.tensor(distribution.sample(), dtype=torch.float32)
    policy.set_weights(weights_init)

    # train
    nes = CoreNES(policy, mdp, alg=alg, optimizer=torch.optim.Adam, optimizer_lr=optim_lr,
                    n_step=(n_epochs), seed=seed)

    nes.train()

    # convert to plot format
    init_params['ep_per_fit'] = population_size * n_rollout

    # logging
    dump_dict = dict({
        'returns_mean': nes.rewards_mean,
        'gain_lqr': gain_lqr,
        'optimal_reward': optimal_reward,
        'best_reward': nes.best_reward,
        'init_params': init_params,
        'alg': alg,
        'ineff_params': ineff_params
    })

    save_results(dump_dict, results_dir, alg, init_params, seed)

    
