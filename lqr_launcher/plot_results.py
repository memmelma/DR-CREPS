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
            key = '|'.join(root.split('/')[-3:])
            data_dict_all[key] = data_dict

    return data_dict_all

def plot_data(data_dict, exp_name, pdf=False):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)

    fig, ax = plt.subplots()
    
    for exp in data_dict.keys():
        y = np.array(data_dict[exp]['returns_mean']).mean(axis=0)
        x = np.arange(0, y.shape[0], 1)
        ci = np.array(data_dict[exp]['returns_mean']).std(axis=0)*2

        ax.plot(x,y, label=exp)
        ax.fill_between(x, (y-ci), (y+ci), alpha=.3)

    # ax.set_ylim(-1000,0)
    # ax.set_xlim(0,10)
    
    ax.set_xlabel('epochs')
    ax.set_ylabel('total return')
    
    plt.hlines(np.array(data_dict[exp]['optimal_reward'][0]).mean(), 0, len(x), 'red', label='optimal control')
    
    ax.legend(prop={'size': 8})

    plt.title(f"{exp_name}\nsamples {data_dict[exp]['init_params'][0]['ep_per_fit']} lqr_dim {data_dict[exp]['init_params'][0]['lqr_dim']} n_ineff {data_dict[exp]['init_params'][0]['n_ineff']}")
    plt.savefig(f"imgs/{exp_name}/returns.{'pdf' if pdf else 'png'}")

def plot_mi(data_dict, exp_name, pdf=False):

    for e, exp in enumerate(data_dict.keys()):
        
        if 'mi_avg' not in data_dict[exp].keys():
            continue
        if not data_dict[exp]['mi_avg'][0]:
            continue

        os.makedirs(f'imgs/{exp_name}', exist_ok=True)
        
        fig, ax = plt.subplots()

        y = np.array(data_dict[exp]['mi_avg']).mean(axis=0)
        x = np.arange(0, y.shape[0], 1)/data_dict[exp]['init_params'][0]['fit_per_epoch']
        ci = np.array(data_dict[exp]['mi_avg']).std(axis=0)*2

        for i in range(y.shape[1]):
            ax.plot(x,y[:,i])
            # ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

        ax.set_xlabel('epochs')
        ax.set_ylabel('average mutual information')

        # ax.legend()

        plt.title(f"{exp_name}\n{exp}")
        plt.savefig(f"imgs/{exp_name}/mi_{e}.{'pdf' if pdf else 'png'}")

def plot_kl(data_dir, exp_name, pdf=False):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
        
    fig, ax = plt.subplots()

    for exp in data_dict.keys():    
        
        y = np.array(data_dict[exp]['kls']).mean(axis=0)
        
        x = np.arange(0, y.shape[0], 1)/data_dict[exp]['init_params'][0]['fit_per_epoch']
        ci = np.array(data_dict[exp]['kls']).std(axis=0)*2

        ax.plot(x,y, label=exp)

    plt.hlines(data_dict[exp]['init_params'][0]['eps'], 0, len(x), 'red', label='KL bound')

    ax.set_xlabel('epochs')
    ax.set_ylabel('average KL')

    ax.legend(prop={'size': 8})

    plt.title(f"{exp_name}\nsamples {data_dict[exp]['init_params'][0]['ep_per_fit']}")
    plt.savefig(f"imgs/{exp_name}/kl.{'pdf' if pdf else 'png'}")


if __name__ == '__main__':

    pdf = False
    
    exp_name = 'lqr_dim_10_eff_3_env_0_con_sample_all'
    
    data_dir = os.path.join('logs', exp_name)
    data_dict = load_data_from_dir(data_dir)

    plot_data(data_dict, exp_name, pdf=pdf)

    plot_mi(data_dict, exp_name, pdf=pdf)
    plot_kl(data_dict, exp_name, pdf=pdf)
    