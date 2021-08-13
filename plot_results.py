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
            # key = '|'.join(root.split('/')[-9:])
            key = '|'.join(root.split('/'))
            # add exp to dict
            data_dict_all[key] = data_dict
    return data_dict_all

def plot_data(data_dict, exp_name, episodes=1000, x_axis='samples', n_samples = 1, pdf=False):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
    fig, ax = plt.subplots()
    max_reward = -np.inf
    max_reward_exp = None
    max_y_discount = -np.inf
    max_y_discount_exp = None

    for exp in data_dict.keys():
        
        # CATCHES missing keys, etc.
        # catch failed experiments
        if 'init_params' not in data_dict[exp].keys():
            print(f'{exp} failed')
            continue
        
        # catch deprecation
        if 'method' not in data_dict[exp]['init_params'][0].keys():
            data_dict[exp]['init_params'][0]['method'] = None

        # if data_dict[exp]['init_params'][0]['alg'] == 'REPS':
        #     continue
        # if data_dict[exp]['init_params'][0]['alg'] == 'REPS':
        #     continue
        
        # if data_dict[exp]['init_params'][0]['alg'] == 'REPS':
        #     continue
        
        # # for catching exps w/ some failed runs
        # if len(data_dict[exp]['returns_mean']) < 25:
        #     continue

        # if len(data_dict[exp]['returns_mean'][0]) < data_dict[exp]['init_params'][0]['n_epochs']:
        #     print('n_epochs smaller',exp)

        # for broken bullet runs
        if 'bullet' in exp:
            min_length = np.min([len(x) for x in data_dict[exp]['returns_mean']])
            data_dict[exp]['returns_mean'] = [x[:min_length] for x in data_dict[exp]['returns_mean']]
        
        try:
            y = np.array(data_dict[exp]['returns_mean']).mean(axis=0)[:episodes]
            
            # FILTERS on mean
            # if data_dict[exp]['init_params'][0]['alg'] not in ['ConstrainedREPS', 'ConstrainedREPSMI', 'ConstrainedREPSMIFull']:
            #     continue
            # if np.max(y) < 35:
            #     continue
            # if np.max(y) < 35 and data_dict[exp]['init_params'][0]['alg'] not in ['REPS', 'REPS_MI_full']:
            #     continue
            # if data_dict[exp]['init_params'][0]['alg'] not in ['REPS_MI']:
            #     continue
        
        except:
            print(f'{exp} failed')
            continue

        # determine x axis
        if x_axis == 'samples':
            x = np.arange(0, y.shape[0], 1) * ( data_dict[exp]['init_params'][0]['ep_per_fit'] // n_samples )
            min_len = np.min([len(x) for x in data_dict[exp]['returns_mean']])
            data_dict[exp]['returns_mean'] = [x[:min_len] for x in data_dict[exp]['returns_mean']]
        else: # epochs
            x = np.arange(0, y.shape[0], 1)
            n_samples = 1

        # # plot single experiment
        init_params = data_dict[exp]['init_params'][0]
        params = exp.split('|')[2:]
        params.remove('alg_'+data_dict[exp]['init_params'][0]['alg'])
        label = f"{init_params['alg']} {sorted(params)}"
        ci = (np.array(data_dict[exp]['returns_mean']).std(axis=0)*2)[:episodes]
        
        ax.plot(x*n_samples,y, label=label, linewidth=1)
        ax.fill_between(x*n_samples, (y-ci), (y+ci), alpha=.3)

        # maximum reward
        if np.max(y) > max_reward:
            max_reward = np.max(y)
            max_reward_exp = exp
        print(data_dict[exp]['init_params'][0]['alg'], 'max reward',  np.max(y), 'k', data_dict[exp]['init_params'][0]['k'], 'gamma', data_dict[exp]['init_params'][0]['gamma'])

        # maximum discounted reward
        discount_factor = 0.99
        discount = np.arange(1, y.shape[0]+1, 1)**discount_factor
        discount_reward = np.sum( y * discount )
        if discount_reward > max_y_discount:
            max_y_discount = discount_reward
            max_y_discount_exp = exp
    
    print(f"MAX REWARD\n {data_dict[max_reward_exp]['init_params'][0]['alg']} | eps: {data_dict[max_reward_exp]['init_params'][0]['eps']} |" +
            f"kappa: {data_dict[max_reward_exp]['init_params'][0]['kappa']} | max reward: {max_reward} | k: {data_dict[max_reward_exp]['init_params'][0]['k']} | " +
            f"method: {data_dict[max_reward_exp]['init_params'][0]['method']} | sample_type: {data_dict[max_reward_exp]['init_params'][0]['sample_type']} | " +
            f"gamma {data_dict[max_reward_exp]['init_params'][0]['gamma']}")

    print(f"MAX DISC REWARD\n {data_dict[max_y_discount_exp]['init_params'][0]['alg']} | eps: {data_dict[max_y_discount_exp]['init_params'][0]['eps']} |" +
            f"kappa: {data_dict[max_y_discount_exp]['init_params'][0]['kappa']} | max reward: {max_y_discount} | k: {data_dict[max_y_discount_exp]['init_params'][0]['k']} | " +
            f"method: {data_dict[max_y_discount_exp]['init_params'][0]['method']} | sample_type: {data_dict[max_y_discount_exp]['init_params'][0]['sample_type']} | " +
            f"gamma {data_dict[max_y_discount_exp]['init_params'][0]['gamma']}")

    if 'lqr_dim' in data_dict[max_reward_exp]['init_params'][0].keys():
        plt.title(f"{exp_name}\nsamples {data_dict[max_reward_exp]['init_params'][0]['ep_per_fit']} lqr_dim {data_dict[max_reward_exp]['init_params'][0]['lqr_dim']} n_ineff {data_dict[max_reward_exp]['init_params'][0]['n_ineff']}")
    if 'optimal_reward' in data_dict[max_reward_exp].keys():
        plt.hlines(np.array(data_dict[max_reward_exp]['optimal_reward'][0]).mean(), 0, n_samples*x.max(), 'red', label='optimal control')

    ax.set_xlabel(x_axis)
    ax.set_ylabel('J')

    ax.legend(loc='upper left', bbox_to_anchor=(-0.2,-0.2), prop={'size': 6}, ncol=1)
    # plt.xlim(0, 6000)
    # plt.ylim(-10, 0)
    plt.title(exp_name)
    plt.tight_layout()
    plt.savefig(f"imgs/{exp_name}/returns.{'pdf' if pdf else 'png'}")
    plt.close()

    ## reward for bullet
    if 'bullet' in exp:
        fig, ax = plt.subplots()
        for exp in data_dict.keys():
            if 'reward_mean' in data_dict[exp].keys():
                min_length = np.min([len(x) for x in data_dict[exp]['reward_mean']])
                data_dict[exp]['reward_mean'] = [x[:min_length] for x in data_dict[exp]['reward_mean']]
                
                try:
                    y = np.array(data_dict[exp]['reward_mean']).mean(axis=0)[:episodes]
                except:
                    print(f'{exp} failed')
                    continue

                # determine x axis
                if x_axis == 'samples':
                    x = np.arange(0, y.shape[0], 1) * ( data_dict[exp]['init_params'][0]['ep_per_fit'] // n_samples )
                    min_len = np.min([len(x) for x in data_dict[exp]['reward_mean']])
                    data_dict[exp]['reward_mean'] = [x[:min_len] for x in data_dict[exp]['reward_mean']]
                else:
                    x = np.arange(0, y.shape[0], 1)
                    n_samples = 1

                # plot single experiment
                init_params = data_dict[exp]['init_params'][0]
                params = exp.split('|')[2:]
                params.remove('alg_'+data_dict[exp]['init_params'][0]['alg'])
                label = f"{init_params['alg']} {sorted(params)}"
                ci = (np.array(data_dict[exp]['reward_mean']).std(axis=0)*2)[:episodes]
                ax.plot(x*n_samples,y, label=label, linewidth=1)
                ax.fill_between(x*n_samples, (y-ci), (y+ci), alpha=.3)
        
        ax.set_xlabel(x_axis)
        ax.set_ylabel('R')
        
        ax.legend(loc='upper left', bbox_to_anchor=(-0.1,-0.2), prop={'size': 6}, ncol=1)
        plt.title(exp_name)
        plt.tight_layout()
        plt.savefig(f"imgs/{exp_name}/reward.{'pdf' if pdf else 'png'}")
        plt.close()


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

    ax.legend(loc='upper left', bbox_to_anchor=(-0.2,-0.2), prop={'size': 6}, ncol=1)
    plt.tight_layout()

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

    ax.legend(loc='upper left', bbox_to_anchor=(-0.2,-0.2), prop={'size': 6}, ncol=1)
    plt.tight_layout()

    plt.title(f"{exp_name}\nsamples {data_dict[exp]['init_params'][0]['ep_per_fit']}")
    plt.savefig(f"imgs/{exp_name}/entropy.{'pdf' if pdf else 'png'}")

if __name__ == '__main__':

    pdf = False
    
    exp_name = 'oracle_fixed'
    exp_name = 'bullet_reps_mi'
    # exp_name = 'ship_3_tiles/alg_REPS'
    # exp_name = 'ship_3_tiles/alg_ConstrainedREPS'
    # exp_name = 'ship_3_tiles/alg_RWR'
    # exp_name = 'ship_3_tiles/alg_ConstrainedREPSMI'
    exp_name = 'ship_3_tiles/alg_REPS_MI'
    exp_name = 'bullet_hopper'
    # exp_name = 'bullet_sigma_3e-1'
    # exp_name = 'bullet_walker'
    
    data_dir = os.path.join('logs', exp_name)
    data_dict = load_data_from_dir(data_dir)

    plot_data(data_dict, exp_name, episodes=250, x_axis='samples', pdf=pdf)

    plot_mi(data_dict, exp_name, pdf=pdf)
    plot_kl(data_dict, exp_name, pdf=pdf)
    plot_entropy(data_dict, exp_name, pdf=pdf)
    