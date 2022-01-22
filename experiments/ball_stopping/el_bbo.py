import os
import numpy as np

from environments.ball_rolling_gym_env import BallRollingGym

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from policy.promp_policy import ProMPPolicy

from experiments.utils import init_distribution, init_policy_search_algorithm, save_results

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

    # parametric policy
    policy = ProMPPolicy(n_basis=n_basis, basis_width=1e-3, maxSteps=horizon, output=mdp.info.action_space.shape)

    # search distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, 
                                        sample_strat=sample_strat, lambd=lambd, distribution_class=distribution)

    # policy search algorithm
    alg, params = init_policy_search_algorithm(algorithm_class=alg, params=init_params)
    agent = alg(mdp.info, distribution, policy, features=None, **params)

    # train
    core = Core(agent, mdp)

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
    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'gain_policy': policy.get_weights(),
        'best_reward': np.array(returns_mean).max(),
        'init_params': init_params,
        'alg': alg
    })

    dump_state = dict({
        'distribution': distribution
    })

    save_results(dump_dict, results_dir, alg, init_params, seed)
    alg.__name__ = alg.__name__ + '_state'
    save_results(dump_state, results_dir, alg, init_params, seed)
