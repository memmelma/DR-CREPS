import os
import numpy as np
from scipy.optimize import minimize

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor

from environments import generate_red_LQR
from experiments import init_distribution, init_policy_search_algorithm, save_results

def eval(weights, core, mdp, verbose):
    core.agent.distribution._mu = weights
    core.ctr += 1
    dataset_eval = core.evaluate(n_episodes=1, quiet=0)
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)[0]
    
    if core.ctr % 50 == 0 and not verbose:
        print('sample', core.ctr, 'return', J)

    core.rewards += [J]
    return -J

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

    # parametric policy
    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(mu=approximator)
    

    # sample initial weights
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, 
                                        sample_strat=None, lambd=0, distribution_class='diag')
    weights_init = distribution.sample()

    # dummy policy
    distribution = init_distribution(mu_init=0, sigma_init=0, size=policy.weights_size, 
                                        sample_strat=None, lambd=0, distribution_class='diag')
    alg_tmp, params = init_policy_search_algorithm(algorithm_class='REPS', params=init_params)
    agent = alg_tmp(mdp.info, distribution, policy, **params)
    
    core = Core(agent, mdp)
    core.rewards = []
    core.ctr = 0
    
    # optimize
    if alg == 'NM':
        res = minimize(eval, weights_init, args=(core, mdp, verbose), method='nelder-mead', options=dict({'maxiter': n_epochs*ep_per_fit}))
    elif alg =='BFGS':
        res = minimize(eval, weights_init, args=(core, mdp, verbose), method='L-BFGS-B', options=dict({'maxfun': n_epochs*ep_per_fit}))
    else:
        res = None

    best_weights = res.x
    print('J at end : ' + str(eval(best_weights, core, mdp, verbose))) # add [:-1] if final evaluation

    # logging
    dump_dict = dict({
        'returns_mean': core.rewards[:-1],
        'returns_std': np.zeros_like(core.rewards[:-1]),
        'agent': agent,
        'gain_lqr': gain_lqr,
        'optimal_reward': optimal_reward,
        'best_reward': max(core.rewards),
        'init_params': init_params,
        'alg': alg_tmp,
        'ineff_params': ineff_params
    })

    save_results(dump_dict, results_dir, alg_tmp, init_params, seed)

