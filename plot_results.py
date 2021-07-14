import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_data_from_dir(data_dir):

    data_dict_all = dict()
    
    for root, dirs, files in os.walk(data_dir):
        if not dirs:
            data_dict = dict()
            for file in files:
                if '.' not in file:
                    data_load = joblib.load(os.path.join(root, file))   
                    for key in data_load.keys():
                        if key not in data_dict.keys():
                            data_dict[key] = [data_load[key]]
                        else:
                            data_dict[key] += [data_load[key]]
            key = '|'.join(root.split('/')[-5:])
            data_dict_all[key] = data_dict

    return data_dict_all

def plot_data(data_dict, exp_name, episodes=1000, pdf=False):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)

    fig, ax = plt.subplots()

    max_reward = -np.inf
    max_reward_exp = None
    min_regret = np.inf
    min_regret_exp = None

    for exp in data_dict.keys():
        
        # catch KL violations
        try:
            if np.max(data_dict[exp]['kls']) > 10.:
                continue
        except:
            pass

        # print(exp)
        # if data_dict[exp]['init_params'][0]['k'] != 15:
        #     continue
        print(exp)
        y = np.array(data_dict[exp]['returns_mean']).mean(axis=0)[:episodes]
        if np.max(y) > max_reward:
            max_reward = np.max(y)
            max_reward_exp = exp
        if np.sum(np.max(y) - y) < min_regret:
            min_regret = np.sum(np.max(y) - y)
            min_regret_exp = exp
        
        print(data_dict[exp]['init_params'][0]['alg'], 'min regret',  np.sum(np.max(y) - y), 'max reward',  np.max(y))

        x = np.arange(0, y.shape[0], 1)
        ci = (np.array(data_dict[exp]['returns_mean']).std(axis=0)*2)[:episodes]

        ax.plot(x,y, label=exp, linewidth=1)
        ax.fill_between(x, (y-ci), (y+ci), alpha=.1)
    
    ax.set_xlabel('epochs')
    ax.set_ylabel('total return')
    
    if 'optimal_reward' in data_dict[exp].keys():
        plt.hlines(np.array(data_dict[exp]['optimal_reward'][0]).mean(), 0, len(x), 'red', label='optimal control')
    
    print(f"Best eps: {data_dict[max_reward_exp]['init_params'][0]['eps']} | best kappa: {data_dict[max_reward_exp]['init_params'][0]['kappa']} | max reward: {max_reward}")
    print(f"Best k: {data_dict[max_reward_exp]['init_params'][0]['k']}")
    print(f"Best sample_type: {data_dict[max_reward_exp]['init_params'][0]['sample_type']} gamma {data_dict[max_reward_exp]['init_params'][0]['gamma']}")
    
    print(f"Best eps: {data_dict[min_regret_exp]['init_params'][0]['eps']} | best kappa: {data_dict[min_regret_exp]['init_params'][0]['kappa']} | min regret: {min_regret}")
    print(f"Best k: {data_dict[min_regret_exp]['init_params'][0]['k']}")
    print(f"Best sample_type: {data_dict[min_regret_exp]['init_params'][0]['sample_type']} gamma {data_dict[min_regret_exp]['init_params'][0]['gamma']}")
    
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    # ax.legend(prop={'size': 8})

    # ax.set_ylim(-3, -1)
    # ax.set_xlim(0, 100)

    ax.legend(loc='upper left', bbox_to_anchor=(0.,-0.2), prop={'size': 8}, ncol=1)
    plt.tight_layout()
    if hasattr(data_dict[exp]['init_params'][0],'lqr_dim'):
        plt.title(f"{exp_name}\nsamples {data_dict[exp]['init_params'][0]['ep_per_fit']} lqr_dim {data_dict[exp]['init_params'][0]['lqr_dim']} n_ineff {data_dict[exp]['init_params'][0]['n_ineff']}")
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
        
        # catch KL violations
        try:
            if np.max(data_dict[exp]['kls']) > 10.:
                continue
        except:
            pass

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
    
    exp_name = 'lqr_dim_10_eff_3_env_0_hp_sample/alg_REPS_MI'
    exp_name = 'lqr_dim_10_eff_3_env_0_hp_sample/alg_ConstrainedREPSMI'
    data_dir = os.path.join('logs', exp_name)
    data_dict = load_data_from_dir(data_dir)

    # plot_data(data_dict, exp_name, episodes=250, pdf=pdf)
    plot_data(data_dict, exp_name, episodes=250, pdf=pdf)

    plot_mi(data_dict, exp_name, pdf=pdf)
    plot_kl(data_dict, exp_name, pdf=pdf)
    plot_entropy(data_dict, exp_name, pdf=pdf)
    