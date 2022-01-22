import os
import numpy as np

from mushroom_rl.environments import ShipSteering

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from experiments import init_grad_agent, save_results

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
    
    # init gradient agent
    agent, alg = init_grad_agent(mdp, alg, actor_lr, critic_lr, max_kl, optim_eps, nn_policy=nn_policy)

    core = Core(agent, mdp, 
                preprocessors=[StandardizationPreprocessor(mdp_info=mdp.info)] if nn_policy else None)
    
    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=verbose)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))
    
    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                n_episodes_per_fit=ep_per_fit, quiet=verbose)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=verbose)
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))

        returns_mean += [np.mean(J)]
        returns_std += [np.std(J)]

    # logging
    best_reward = np.array(returns_mean).max()
    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'agent': agent,
        'optimal_reward': np.ones_like(best_reward)*-59,
        'best_reward': best_reward,
        'init_params': init_params,
        'alg': alg
    })

    save_results(dump_dict, results_dir, alg, init_params, seed)
    