import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_data_from_dir(data_dir):
    # create dict for all exps
    data_dict_all = dict()
    # walk folders
    for root, dirs, files in os.walk(data_dir):
        # if file is in folder
        if not dirs:
            # make dict for exp
            data_dict = dict()
            for file in files:
                # exclude slurm.sh
                if '.' not in file:
                    # load file
                    data_load = joblib.load(os.path.join(root, file))   
                    # add exp to dict
                    for key in data_load.keys():
                        # create new key for exp
                        if key not in data_dict.keys():
                            data_dict[key] = [data_load[key]]
                        # add run to exp
                        else:
                            data_dict[key] += [data_load[key]]
            # set key for all runs of exp
            key = '|'.join(root.split('/')[-8:])
            # add exp to dict
            data_dict_all[key] = data_dict
    return data_dict_all

def plot_data(data_dict, exp_name, episodes=1000, x_axis='samples', pdf=False):

    n_samples = 25
    
    os.makedirs(f'imgs/{exp_name}', exist_ok=True)

    fig, ax = plt.subplots()

    max_reward = -np.inf
    max_reward_exp = None
    min_regret = np.inf
    min_regret_exp = None

    max_y_discount = -np.inf

    # plt.xlim(0, 20000)
    # plt.ylim(-10, 15)
    for exp in data_dict.keys():
        
        if 'init_params' not in data_dict[exp].keys():
            print(f'{exp} failed')
            continue
        
        if len(data_dict[exp]['init_params']) < 5:
            continue

        if 'method' not in data_dict[exp]['init_params'][0].keys():
            data_dict[exp]['init_params'][0]['method'] = None

        if x_axis != 'samples':
            n_samples = data_dict[exp]['init_params'][0]['ep_per_fit']

        # if 'REPS' == data_dict[exp]['init_params'][0]['alg'] or 'REPS_MI' == data_dict[exp]['init_params'][0]['alg']:
        #     print(data_dict[exp]['init_params'][0]['alg'])
        # else:
        #     continue
        # if data_dict[exp]['init_params'][0]['alg'] != 'MORE':
        #     print(f'{exp} skipped')
        #     continue
        
        # if data_dict[exp]['init_params'][0]['kappa'] > 10:
        #     print(f'{exp} kappa too large')
        #     continue

        # # catch KL violations
        # try:
        #     if np.max(data_dict[exp]['kls']) > 10.:
        #         print(f'kl {np.max(data_dict[exp]["kls"])} detected')
        #         continue
        # except:
        #     pass
        # try:
        #     if data_dict[exp]['init_params'][0]['kappa'] == 10000. or data_dict[exp]['init_params'][0]['ep_per_fit'] == 150 or data_dict[exp]['init_params'][0]['ep_per_fit'] == 25 or data_dict[exp]['init_params'][0]['eps'] == 1.0:
        #         print(f'entropy == 10000 detected')
        #         continue
        # except:
        #     pass

        min_len = np.min([len(x) for x in data_dict[exp]['returns_mean']])
        data_dict[exp]['returns_mean'] = [x[:min_len] for x in data_dict[exp]['returns_mean']]

        y = np.array(data_dict[exp]['returns_mean']).mean(axis=0)[:episodes]

        discount_factor = 0.99
        discount = np.arange(1, y.shape[0]+1, 1)**discount_factor

        discount_reward = np.sum( y * discount )

        if discount_reward > max_y_discount:
            max_discount_reward = discount_reward
            max_discount_reward_exp = exp
        if np.max(y) > max_reward:
            max_reward = np.max(y)
            max_reward_exp = exp
        if np.sum(np.max(y) - y) < min_regret:
            min_regret = np.sum(np.max(y) - y)
            min_regret_exp = exp
        
        print(data_dict[exp]['init_params'][0]['alg'], 'min regret',  np.sum(np.max(y) - y), 'max reward',  np.max(y))

        x = np.arange(0, y.shape[0], 1) * ( data_dict[exp]['init_params'][0]['ep_per_fit'] // n_samples )
        # x = np.arange(0, y.shape[0], 1)
        ci = (np.array(data_dict[exp]['returns_mean']).std(axis=0)*2)[:episodes]

        ax.plot(x*n_samples,y, label=exp, linewidth=1)
        ax.fill_between(x*n_samples, (y-ci), (y+ci), alpha=.1)
        
    ax.set_xlabel(x_axis)
    ax.set_ylabel('total return')
    
    if 'optimal_reward' in data_dict[exp].keys():
        plt.hlines(np.array(data_dict[exp]['optimal_reward'][0]).mean(), 0, n_samples*x.max(), 'red', label='optimal control')
    
    print(f"Max reward: {data_dict[max_reward_exp]['init_params'][0]['alg']}")
    print(f"Best eps: {data_dict[max_reward_exp]['init_params'][0]['eps']} | best kappa: {data_dict[max_reward_exp]['init_params'][0]['kappa']} | max reward: {max_reward}")
    print(f"Best k: {data_dict[max_reward_exp]['init_params'][0]['k']}")
    print(f"Best method: {data_dict[max_reward_exp]['init_params'][0]['method']}")
    print(f"Best sample_type: {data_dict[max_reward_exp]['init_params'][0]['sample_type']} gamma {data_dict[max_reward_exp]['init_params'][0]['gamma']}")
    
    print(f"Min regret: {data_dict[min_regret_exp]['init_params'][0]['alg']}")
    print(f"Best eps: {data_dict[min_regret_exp]['init_params'][0]['eps']} | best kappa: {data_dict[min_regret_exp]['init_params'][0]['kappa']} | min regret: {min_regret}")
    print(f"Best k: {data_dict[min_regret_exp]['init_params'][0]['k']}")
    # print(f"Best method: {data_dict[min_regret_exp]['init_params'][0]['method']}")
    print(f"Best sample_type: {data_dict[min_regret_exp]['init_params'][0]['sample_type']} gamma {data_dict[min_regret_exp]['init_params'][0]['gamma']}")
    
    print(f"Max disc. reward: {data_dict[max_discount_reward_exp]['init_params'][0]['alg']}")
    print(f"Best eps: {data_dict[max_discount_reward_exp]['init_params'][0]['eps']} | best kappa: {data_dict[max_discount_reward_exp]['init_params'][0]['kappa']} | max_discount_reward: {max_discount_reward}")
    print(f"Best k: {data_dict[max_discount_reward_exp]['init_params'][0]['k']}")
    # print(f"Best method: {data_dict[max_discount_reward_exp]['init_params'][0]['method']}")
    print(f"Best sample_type: {data_dict[max_discount_reward_exp]['init_params'][0]['sample_type']} gamma {data_dict[max_discount_reward_exp]['init_params'][0]['gamma']}")

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    # ax.legend(prop={'size': 8})

    # ax.set_ylim(-3, -1)
    # ax.set_xlim(0, 100)

    # ax.legend(loc='upper left', bbox_to_anchor=(0.,-0.2), prop={'size': 6}, ncol=1)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.2,-0.2), prop={'size': 6}, ncol=1)
    if 'lqr_dim' in data_dict[exp]['init_params'][0].keys():
        plt.title(f"{exp_name}\nsamples {data_dict[exp]['init_params'][0]['ep_per_fit']} lqr_dim {data_dict[exp]['init_params'][0]['lqr_dim']} n_ineff {data_dict[exp]['init_params'][0]['n_ineff']}")
    plt.tight_layout()
    plt.savefig(f"imgs/{exp_name}/returns.{'pdf' if pdf else 'png'}")

def plot_mi(data_dict, exp_name, pdf=False):
    
    for e, exp in enumerate(data_dict.keys()):
        
        # catch KL violations
        try:
            if np.max(data_dict[exp]['kls']) > 10.:
                continue
        except:
            pass

        if 'mi_avg' not in data_dict[exp].keys():
            continue
        if data_dict[exp]['mi_avg'][0] is None:
            continue

        os.makedirs(f'imgs/{exp_name}', exist_ok=True)
        
        fig, ax = plt.subplots()

        y = np.array(data_dict[exp]['mi_avg']).mean(axis=0)
        x = np.arange(0, y.shape[0], 1)/data_dict[exp]['init_params'][0]['fit_per_epoch']
        ci = np.array(data_dict[exp]['mi_avg']).std(axis=0)*2

        # ax.plot(x,y)
        # print('REMOVE FOR STANDARD BEHAVIOR')
        for i in range(y.shape[1]):
            ax.plot(x,y[:,i])
            ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

        ax.set_xlabel('epochs')
        ax.set_ylabel('average mutual information')

        # ax.legend()

        plt.title(f"{exp_name}\n{exp}")
        plt.savefig(f"imgs/{exp_name}/mi_{e}.{'pdf' if pdf else 'png'}")

def plot_kl(data_dir, exp_name, pdf=False):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
        
    fig, ax = plt.subplots()

    for exp in data_dict.keys():    
        
        # # catch KL violations
        # try:
        #     if np.max(data_dict[exp]['kls']) > 10.:
        #         continue
        # except:
        #     pass

        if 'kls' not in data_dict[exp].keys():
            continue
        if data_dict[exp]['kls'][0] is None:
            continue

        y = np.array(data_dict[exp]['kls']).mean(axis=0)
        
        x = np.arange(0, y.shape[0], 1)/data_dict[exp]['init_params'][0]['fit_per_epoch']
        ci = np.array(data_dict[exp]['kls']).std(axis=0)*2

        ax.plot(x.squeeze(),y.squeeze(), label=exp)

    plt.hlines(data_dict[exp]['init_params'][0]['eps'], 0, len(x), 'red', label='KL bound')

    ax.set_xlabel('epochs')
    ax.set_ylabel('average KL')

    ax.legend(prop={'size': 8})

    plt.title(f"{exp_name}\nsamples {data_dict[exp]['init_params'][0]['ep_per_fit']}")
    plt.savefig(f"imgs/{exp_name}/kl.{'pdf' if pdf else 'png'}")

def plot_entropy(data_dir, exp_name, pdf=False):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
        
    fig, ax = plt.subplots()

    for exp in data_dict.keys():    
        
        # catch KL violations
        try:
            if np.max(data_dict[exp]['kls']) > 10.:
                continue
        except:
            pass

        if 'entropys' not in data_dict[exp].keys():
            continue
        if data_dict[exp]['entropys'][0] is None:
            continue

        y = np.array(data_dict[exp]['entropys']).mean(axis=0)
        
        x = np.arange(0, y.shape[0], 1)/data_dict[exp]['init_params'][0]['fit_per_epoch']
        ci = np.array(data_dict[exp]['entropys']).std(axis=0)*2

        ax.plot(x,y, label=exp)

    plt.hlines(data_dict[exp]['init_params'][0]['kappa'], 0, len(x), 'red', label='entropy bound')

    ax.set_xlabel('epochs')
    ax.set_ylabel('average entropy')

    ax.legend(prop={'size': 8})

    plt.title(f"{exp_name}\nsamples {data_dict[exp]['init_params'][0]['ep_per_fit']}")
    plt.savefig(f"imgs/{exp_name}/entropy.{'pdf' if pdf else 'png'}")

if __name__ == '__main__':

    pdf = False
    
    # exp_name = 'lqr_dim_10_eff_3_env_0_hp_sample/alg_REPS_MI'
    # exp_name = 'lqr_dim_10_eff_3_env_0_hp_sample/alg_ConstrainedREPSMI'
    # exp_name = 'all_best'
    # exp_name = 'lqr_dim_10_eff_3_env_0_more'
    # exp_name = 'con_reps_eps_kappa'
    
    # exp_name = 'lqr_dim_10_eff_3_env_0_sample_type/alg_REPS_MI'
    # exp_name = 'lqr_dim_10_eff_3_env_0_sample_type/alg_ConstrainedREPSMI'

    # exp_name = 'ship_sample_type_bins_3/alg_REPS_MI'
    # exp_name = 'ship_sample_type_bins_3/alg_ConstrainedREPSMI'

    # exp_name = 'ship_sample_type_bins_4_t_3/alg_REPS_MI'
    # exp_name = 'ship_sample_type_bins_4_t_3/alg_ConstrainedREPSMI'

    # exp_name = 'ball/best_eps_kappa'
    # exp_name = 'ball/con_reps_mi_k_gamma'
    # exp_name = 'lqr/all_best'
    # exp_name = 'lqr_dim_10_eff_3_env_0_sample_type_solong'
    # exp_name = 'lqr_dim_10_eff_3_env_0_toy_long'
    exp_name = 'ball/pearson_vs_mi_25'
    # exp_name = 'lqr_dim_10_eff_3_env_0_full_cov_25'
    # exp_name = 'ship/all_best'

    data_dir = os.path.join('logs', exp_name)
    data_dict = load_data_from_dir(data_dir)

    plot_data(data_dict, exp_name, episodes=250, x_axis='samples', pdf=pdf)

    plot_mi(data_dict, exp_name, pdf=pdf)
    plot_kl(data_dict, exp_name, pdf=pdf)
    plot_entropy(data_dict, exp_name, pdf=pdf)
    