import os
import argparse
import joblib
import numpy as np

from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles

from mushroom_rl.environments import ShipSteering

from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor

from utils_el import init_distribution, init_algorithm, log_constraints

def experiment( n_tilings, \
                alg, eps, kappa, k, \
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
    input_shape = (features.size,)

    if 'Oracle' in alg or 'ORACLE' in alg:

        # # get ORACLE
        # oracle_dict = dict()
        # for seed in range(5):
        #     mdp_oracle = ShipSteering()
        #     approximator_oracle = Regressor(LinearApproximator, input_shape=input_shape,
        #                             output_shape=mdp_oracle.info.action_space.shape)
        #     policy_oracle = DeterministicPolicy(approximator_oracle)
            
        #     if tilings == 1:
        #     # path = 'logs/ship/all_best/alg_REPS_MI/k_25/sample_type_percentage/gamma_0.6/eps_0.9'
        #         path = f'logs/ship/all_best/alg_REPS/eps_0.9/'
        #         algo = f'REPS_{seed}'
        #     elif tilings == 3:
        #         path = f'logs/ship/3_tiles/alg_ConstrainedREPSMI/k_55/sample_type_percentage/gamma_0.9/eps_9.5/kappa_12.0/'
        #         algo = f'ConstrainedREPSMI_{seed}'
            
        #     init_params = joblib.load(os.path.join(path, algo))['init_params']
        #     distribution_oracle = joblib.load(os.path.join(path, f'{algo}_state'))['distribution']
        #     alg_oracle, params_orcale = init_algorithm(algorithm_class='ConstrainedREPSMI', params=init_params)
        #     agent_oracle = alg_oracle(mdp_oracle.info, distribution_oracle, policy_oracle, features=features, **params_orcale)
        #     core_oracle = Core(agent_oracle, mdp_oracle)

        #     eval_oracle = core_oracle.evaluate(n_episodes=1, quiet=quiet)
            
        #     oracle = np.arange(0, policy_oracle.weights_size, 1)[agent_oracle.states > 0].tolist()
        #     del policy_oracle, distribution_oracle, alg_oracle, params_orcale, agent_oracle, core_oracle, eval_oracle, approximator_oracle, mdp_oracle
            
        #     # count parameter occurences 
        #     for orac in oracle:
        #         if str(orac) in oracle_dict.keys():
        #             oracle_dict[str(orac)] += 1
        #         else:
        #             oracle_dict[str(orac)] = 1
        #     print('oracle',oracle)

        #     # only select parameters that appear at least 50% of the time in 25 runs 
        #     oracle_new = []
        #     for key in oracle_dict.keys():
        #         val = oracle_dict[key]
        #         # if val > 12:
        #         if val > 0:
        #             oracle_new += [key]
        # print('oracle_all', oracle_new)
        # exit()
        
        if n_tilings == 1:
            oracle = ['75', '80', '81', '86', '87', '92', '93', '100']
            oracle = ['75', '80', '81', '86', '87', '92', '93', '100', '118', '105']
        elif n_tilings == 3:
            oracle = ['80', '81', '86', '87', '93', '100', '105', '112', '117', '118',
                     '225', '230', '231', '236', '237', '242', '243', '250', '380', '386',
                      '392', '393', '400', '405', '92', '106', '111']
            oracle = ['80', '81', '86', '87', '93', '100', '105', '112', '117', '118',
                    '225', '230', '231', '236', '237', '242', '243', '250', '380', '381',
                     '386', '387', '392', '393', '400', '405', '92', '106', '111', '123',
                      '248', '385', '391', '397']
        else:
            print('Define Oracle or specify path! Exiting...')
            exit()
        #     print('Using hard coded Oracle!')
            # print('Failed to load Oracle!')
            # if seed >= 0 and seed < 25 and tilings == 1:
            #     oracle = [75, 80, 81, 86, 87, 92, 93, 100]
            #     print('Using hard coded Oracle!')
            # elif seed >= 0 and seed < 25 and tilings == 3:
            # else:
            #     print('Define Oracle or specify path! Exiting...')
            #     exit()

    else:
        oracle=None

    init_params = locals()
    init_params['oracle'] = oracle

    # init approximator
    approximator = Regressor(LinearApproximator, input_shape=input_shape,
                             output_shape=mdp.info.action_space.shape)
    policy = DeterministicPolicy(approximator)


    # init distribution
    distribution = init_distribution(mu_init=0, sigma_init=sigma_init, size=policy.weights_size, sample_type=sample_type, gamma=gamma, distribution_class=distribution)
    
    print('action space', mdp.info.action_space.shape)
    print('parameters', policy.weights_size)

    # init agent
    alg, params = init_algorithm(algorithm_class=alg, params=init_params)
    agent = alg(mdp.info, distribution, policy, features=features, **params)

    # train
    core = Core(agent, mdp)

    dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet, render=not quiet)
    # print('distribution parameters: ', distribution.get_parameters())
    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    returns_mean = [np.mean(J)]
    returns_std = [np.std(J)]

    for i in range(n_epochs):
        
        core.learn(n_episodes=fit_per_epoch * ep_per_fit,
                   n_episodes_per_fit=ep_per_fit, quiet=quiet)

        dataset_eval = core.evaluate(n_episodes=ep_per_fit, quiet=quiet, render=not quiet)
        # print('distribution parameters: ', distribution.get_parameters())
        J = compute_J(dataset_eval, gamma=mdp.info.gamma)
        print('J at iteration ' + str(i) + ': ' + str(round(np.mean(J),4)))
        
        returns_mean += [np.mean(J)]
        returns_std += [np.std(J)]

    agent.states = np.zeros(policy.weights_size)
    dataset_eval = core.evaluate(n_episodes=1, quiet=quiet, render=not quiet)
    print(agent.states.astype(np.int32))

    # logging
    gain_policy = policy.get_weights()
    mus, kls, entropys, mi_avg = log_constraints(agent)
    best_reward = np.array(returns_mean).max()

    dump_dict = dict({
        'returns_mean': returns_mean,
        'returns_std': returns_std,
        # 'agent': agent,
        'gain_policy': gain_policy,
        'best_reward': best_reward,
        'optimal_reward': np.ones_like(best_reward)*-59,
        'init_params': init_params,
        'alg': alg,
        'mi_avg': mi_avg,
        'mus': mus,
        'kls': kls,
        'entropys': entropys
    })

    joblib.dump(dump_dict, os.path.join(results_dir, f'{alg.__name__}_{seed}'))
    
    dump_state = dict({
        'distribution': distribution
    })

    joblib.dump(dump_state, os.path.join(results_dir, f'{alg.__name__}_{seed}_state'))

    filename = os.path.join(results_dir, f'log_{alg.__name__}_{seed}.txt')
    os.makedirs(results_dir, exist_ok=True)
    with open(filename, 'w') as file:
        for key in init_params.keys():
            file.write(f'{key}: {init_params[key]}\n')


def default_params():
    defaults = dict(
        # environment
        n_tilings = 3,

        # algorithm
        # alg = 'REPS_MI_ORACLE',
        # alg = 'ConstrainedREPSMIOracle',
        # alg = 'ConstrainedREPSMI',
        # alg = 'REPS_MI',
        alg = 'REPS',
        eps = 0.9,
        kappa = 14.0,
        k = 8,

        # distribution
        sigma_init = 7e-2,
        # distribution = 'diag',
        distribution = 'cholesky',

        # MI related
        method = 'MI', # Pearson
        mi_type = 'regression',
        bins = 4,
        sample_type = None,
        gamma = 0.9,
        mi_avg = 0, # False

        # training
        n_epochs = 10,
        fit_per_epoch = 1, 
        ep_per_fit = 500,

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
