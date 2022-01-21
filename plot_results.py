import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

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
            key = '|'.join(root.split('/'))
            data_dict_all[key] = data_dict
    return data_dict_all

def plot_data(data_dict, exp_name, samples=5000, n_samples = 1, max_runs=25, pdf=False):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
    fig, ax = plt.subplots()
    max_reward = - np.inf
    
    for i_exp, exp in enumerate(sorted(data_dict.keys(), reverse=True)):

        if 'init_params' not in data_dict[exp].keys():
            print(f'{exp} failed')
            continue
        if len(data_dict[exp]['returns_mean']) < max_runs:
            continue
        
        y, ci  = get_mean_and_confidence(np.array(data_dict[exp]['returns_mean']))
        ci = ci[1]

        if samples > 0:
            cut = samples // data_dict[exp]['init_params'][0]['ep_per_fit']
            x = np.arange(0, cut, 1) * ( data_dict[exp]['init_params'][0]['ep_per_fit'] // n_samples )
            y = y[:cut]
        else:
            cut = None
            x = np.arange(0, y.shape[0], 1) * ( data_dict[exp]['init_params'][0]['ep_per_fit'] // n_samples )

        min_len = np.min([len(x) for x in data_dict[exp]['returns_mean']])
        data_dict[exp]['returns_mean'] = [x[:min_len] for x in data_dict[exp]['returns_mean']]

        # # plot single experiment
        init_params = data_dict[exp]['init_params'][0]
        params = exp.split('|')[2:]
        params.remove('alg_'+data_dict[exp]['init_params'][0]['alg'])
    
        label = f"{init_params['alg']} {sorted(params)}"

        ax.plot(x*n_samples,y, label=label, linewidth=2)
        ax.fill_between(x*n_samples, (y-ci), (y+ci), alpha=.3)
            
        # maximum reward
        if np.max(y) > max_reward:
            max_reward = np.max(y)
            max_reward_exp = exp


        print(data_dict[exp]['init_params'][0]['alg'], np.round(np.max(y), 4), sorted(params))
        print('completed runs',len(data_dict[exp]['returns_mean']))

    print(f"MAX REWARD\n {data_dict[max_reward_exp]['init_params'][0]['alg']} | eps: {data_dict[max_reward_exp]['init_params'][0]['eps']} |" +
            f"kappa: {data_dict[max_reward_exp]['init_params'][0]['kappa']} | max reward: {max_reward} | k: {data_dict[max_reward_exp]['init_params'][0]['k']} | " +
            f"C: {data_dict[max_reward_exp]['init_params'][0]['C']} | sample_strat: {data_dict[max_reward_exp]['init_params'][0]['sample_strat']} | " +
            f"lambd {data_dict[max_reward_exp]['init_params'][0]['lambd']}" + f" | ep_per_fit {data_dict[max_reward_exp]['init_params'][0]['ep_per_fit']}" +
            f" | n_epochs {data_dict[max_reward_exp]['init_params'][0]['n_epochs']}")

    if 'optimal_reward' in data_dict[max_reward_exp].keys():
        # plt.hlines(np.array(data_dict[max_reward_exp]['optimal_reward'][0]).mean(), 0, n_samples*x.max(), 'red', label='optimal control')
        print('OPTIMAL', np.array(data_dict[max_reward_exp]['optimal_reward'][0]).mean())
    
    ax.set_xlabel('episodes', fontsize=20)
    ax.set_ylabel('J', fontsize=20)

    ax.legend(loc='upper left', bbox_to_anchor=(-0.2,-0.2), prop={'size': 6}, ncol=1)

    plt.tight_layout()
    plt.grid()
    if 'lqr_diag' in exp:
        ratio = 10/16
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.savefig(f"imgs/{exp_name}/returns.{'pdf' if pdf else 'png'}", bbox_inches='tight', pad_inches=0)
    plt.close()


def get_mean_and_confidence(data):
    """
    Compute the mean and 95% confidence interval
    Args:
        data (np.ndarray): Array of experiment data of shape (n_runs, nepochs).
    Returns:
        The mean of the dataset at each epoch along with the confidence interval.
    """
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    interval  = st.t.interval(0.95, n - 1, scale=se)
    return mean, interval

if __name__ == '__main__':


    exp_name = 'test'
    max_runs = 5

    data_dir = os.path.join('logs', exp_name)

    data_dict = load_data_from_dir(data_dir)

    plot_data(data_dict, exp_name, episodes=1000, samples=-1, pdf=False, max_runs=max_runs, clean=False)
