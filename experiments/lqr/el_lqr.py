import os
import joblib
import numpy as np

from mushroom_rl.environments import LQR

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor


from experiments import init_distribution, init_algorithm

def experiment_lqr(
    env, seed, env_seed, \
    lqr_dim, red_dim, \
    n_tilings, \
    n_basis, horizon, \
    alg, eps, kappa, k, \
    distribution, sigma_init, \
    C, mi_estimator, \
    sample_strat, lambd, \
    n_epochs, fit_per_epoch, ep_per_fit, \
    results_dir, save_render_path, verbose
):
    
    # misc
    verbose = bool(verbose==0)
    init_params = locals()
    np.random.seed(seed)
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    horizon = 50
    mdp = LQR.generate(dimensions=lqr_dim, horizon=horizon, max_pos=1., max_action=1.)

    rng = np.random.default_rng(seed=0)
    choice = np.arange(0,lqr_dim,1)
    rng.shuffle(choice)

    ineff_params = choice[:red_dim]
    
    for p in ineff_params:
        mdp.B[p][p] = 1e-20
        mdp.Q[p][p] = 1e-20
    
    # compute optimal control return
    gain_lqr = compute_lqr_feedback_gain(mdp, max_iterations=100000)
    state = mdp.reset()
    optimal_reward = 0
    for i in range(horizon):
        action = - gain_lqr @ state
        state, reward, _, __ = mdp.step(action)
        optimal_reward += reward

    # parametric policy
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(mu=approximator)

    # search distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, 
                                        sample_strat=sample_strat, lambd=lambd, distribution_class=distribution)

    # policy search algorithm
    alg, params = init_algorithm(algorithm_class=alg, params=init_params)
    agent = alg(mdp.info, distribution, policy, **params)

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
        'top_k_corr': agent.top_k_corr if hasattr(agent, 'top_k_corr') else None,
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        'agent': agent,
        'gain_lqr': gain_lqr,
        'gain_policy': policy.get_weights(),
        'optimal_reward': optimal_reward,
        'best_reward': np.array(returns_mean).max(),
        'init_params': init_params,
        'alg': alg,
        'ineff_params': ineff_params
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg.__name__}_{seed}'))
    
    filename = os.path.join(results_dir, f'log_{alg.__name__}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')
