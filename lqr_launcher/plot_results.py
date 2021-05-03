import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_data_from_dir(data_dir):

    runs = os.walk(data_dir)
    run = next(runs)[0]

    returns_mean_all = []
    returns_std_all = []
    gain_lqr_all = []
    gain_policy_all = []
    mi_avg_all = []
    mus_all = []
    kls_all = []
    legend = []
    optimal_reward_all = []
    best_reward_all = []

    subdirs = os.listdir(run)
    for subdir in subdirs:
        if '.' not in subdir:
            legend += [subdir]
            files = os.listdir(os.path.join(run, subdir))
            returns_mean = None
            returns_std = None
            gain_lqr = None
            gain_policy = None
            init_params = None
            mi_avg = None
            mus = None
            kls = None
            optimal_reward = None
            best_reward = None

            file_count = 0
            print(os.path.join(run, subdir))
            for file in files:
                
                if '.' not in file:
                    data = joblib.load(os.path.join(run, subdir, file))

                    # dict_keys(['returns_mean', 'returns_std', 'agent', 'gain_lqr', 'gain_policy', 'optimal_reward', 'best_reward', init_params', 'alg', 'mi_avg'])
                    
                    # print(f"{data['alg'].__name__} - optimal: {data['optimal_reward']} - best: {data['best_reward']})")

                    if returns_mean is None:

                        # print(data['alg'].__name__, data.keys())
                        # print(data['ineff_params'])

                        returns_mean =[data['returns_mean']]
                        returns_std = [data['returns_std']]
                        gain_lqr = [data['gain_lqr']]
                        gain_policy = [data['gain_policy']]
                        
                        # print('LQR', list(gain_lqr))
                        # print('Policy', list(gain_policy))

                        optimal_reward = [data['optimal_reward']]
                        best_reward = [data['best_reward']]

                        init_params = data['init_params']
                        
                        if 'MI' in data['alg'].__name__:
                            mi_avg = [data['mi_avg']]

                        mus = [data['mus']]
                        kls = [data['kls']]

                    else:
                        returns_mean += [data['returns_mean']]
                        returns_std += [data['returns_std']]

                        gain_lqr += [data['gain_lqr']]
                        gain_policy += [data['gain_policy']]
                        
                        optimal_reward += [data['optimal_reward']]
                        best_reward += [data['best_reward']]

                        if 'MI' in data['alg'].__name__:
                            mi_avg += [data['mi_avg']]

                        mus += [data['mus']]
                        kls += [data['kls']]

                    file_count += 1
            
            # returns_mean = returns_mean / file_count
            # returns_std = returns_std / file_count
            # gain_lqr = gain_lqr / file_count
            # gain_policy = gain_policy / file_count
            # if data['alg'].__name__ == 'REPS_MI':
            #     mi_avg = mi_avg / file_count

            returns_mean_all += [returns_mean]
            returns_std_all += [returns_std]
            gain_lqr_all += [gain_lqr]
            gain_policy_all += [gain_policy]
            mi_avg_all += [mi_avg]
            mus_all += [mus]
            kls_all += [kls]
            optimal_reward_all += [optimal_reward]
            best_reward_all += [best_reward]

    return returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, mus_all, kls_all, legend, init_params

def load_data_from_dir_segway(data_dir):

    runs = os.walk(data_dir)
    run = next(runs)[0]

    returns_mean_all = []
    returns_std_all = []
    mi_avg_all = []
    legend = []
    best_reward_all = []

    subdirs = os.listdir(run)
    for subdir in subdirs:
        if '.' not in subdir:
            legend += [subdir]
            files = os.listdir(os.path.join(run, subdir))
            returns_mean = None
            returns_std = None
            init_params = None
            mi_avg = None
            best_reward = None

            file_count = 0
            print(os.path.join(run, subdir))
            for file in files:
                
                if '.' not in file:
                    data = joblib.load(os.path.join(run, subdir, file))

                    # dict_keys(['returns_mean', 'returns_std', 'agent', 'gain_lqr', 'gain_policy', 'optimal_reward', 'best_reward', init_params', 'alg', 'mi_avg'])
                    
                    # print(f"{data['alg'].__name__} - optimal: {data['optimal_reward']} - best: {data['best_reward']})")

                    if returns_mean is None:
                        print(data['alg'].__name__, data.keys())
                        returns_mean =[data['returns_mean']]
                        returns_std = [data['returns_std']]
                        best_reward = [data['best_reward']]
                        init_params = data['init_params']
                        if 'MI' in data['alg'].__name__:
                            mi_avg = [data['mi_avg']]

                    else:
                        returns_mean += [data['returns_mean']]
                        returns_std += [data['returns_std']]
                        
                        best_reward += [data['best_reward']]

                        if 'MI' in data['alg'].__name__:
                            mi_avg += [data['mi_avg']]
                    file_count += 1

            returns_mean_all += [returns_mean]
            returns_std_all += [returns_std]
            mi_avg_all += [mi_avg]
            best_reward_all += [best_reward]

    return returns_mean_all, returns_std_all, None, best_reward_all, mi_avg_all, legend, init_params

def plot_data(returns_mean_all, returns_std_all, optimal_reward_all, legend, exp_name, config):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)

    fig, ax = plt.subplots()
    
    colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

    for i in range(len(returns_mean_all)):
        y = np.array(returns_mean_all[i]).mean(axis=0)
        x = np.arange(0, y.shape[0], 1)
        ci = np.array(returns_mean_all[i]).std(axis=0)*2

        # print(np.array(returns_mean_all[i])[:,0])
        # print(np.array(returns_mean_all[i])[:,0].shape)
        # print(np.array(returns_mean_all[i])[:,0].mean(), np.array(returns_mean_all[i])[:,0].std())

        ax.plot(x,y, label=legend[i])#, color=colors[i])
        ax.fill_between(x, (y-ci), (y+ci), alpha=.3)#, color=colors[i])
        # ax.errorbar(x, y, yerr=ci, errorevery=10)

    # ax.set_ylim(-1000,0)
    # ax.set_xlim(0,10)
    
    ax.set_xlabel('epochs')
    ax.set_ylabel('total return')
    
    plt.hlines(np.array(optimal_reward_all[0]).mean(), 0, len(x), 'red', label='optimal control')

    plt.legend()
    # plt.savefig(f'imgs/{exp_name}/{exp_name}.pdf')

    plt.title(f'samples {config["ep_per_fit"]} lqr_dim {init_params["lqr_dim"]} n_ineff {init_params["n_ineff"]}')
    plt.savefig(f'imgs/{exp_name}/{exp_name}.png')

def plot_mi(avg_mi_all, mi_idx, exp_name, config):

    avg_mi_all = np.array(avg_mi_all[mi_idx])
    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
    
    fig, ax = plt.subplots()

    y = avg_mi_all.mean(axis=0)
    # y = np.array(avg_mi_all).mean(axis=0)

    x = np.arange(0, y.shape[0], 1)/config['fit_per_epoch']
    # ci = np.array(avg_mi_all).std(axis=0)
    ci = avg_mi_all.std(axis=0)*2

    for i in range(y.shape[1]):
        ax.plot(x,y[:,i])
        # ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

    ax.set_xlabel('epochs')
    ax.set_ylabel('average mutual information')

    plt.legend(list(range(y.shape[1])))
    # plt.savefig(f'imgs/{exp_name}/{exp_name}_mi.pdf')

    plt.title(f'MI samples {config["ep_per_fit"]} w/ {legend[mi_idx]}')
    # plt.legend(list(range(config['lqr_dim'])))
    plt.savefig(f'imgs/{exp_name}/{exp_name}_mi.png')

def plot_mu(mus_all, exp_name, config):

    mus_all = np.array(mus_all)
    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
    
    fig, ax = plt.subplots()
    y = mus_all.mean(axis=1)
    x = np.arange(0, y.shape[1], 1)/config['fit_per_epoch']
    ci = mus_all.std(axis=1)*2

    legend = []
    linestyles = ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']
    colors = ['red', 'blue', 'green', 'orange']
    algs = ['reps_', 'reps_mi_']
    for j in range(y.shape[0]):
        for i in range(y.shape[2]):
            ax.plot(x,y[j,:,i], linestyle=linestyles[j])#, color=colors[i])
        # legend += [str(algs[j])]
        # ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

    ax.set_xlabel('epochs')
    ax.set_ylabel('average policy mean')

    # plt.title(f'MI samples {config["ep_per_fit"]} w/ {legend[mi_idx]}')
    plt.title(f'REPS : REPS_MI -')
    # plt.legend(legend)
    plt.savefig(f'imgs/{exp_name}/{exp_name}_mus.png')

def plot_mu_diff(mus_all, exp_name, config):

    n_epochs = 900

    mus_all = np.array(mus_all)
    mus_all = np.diff(mus_all, axis=2, prepend=0)[:,:,:n_epochs,:]
    mus_all = np.abs(mus_all)

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
    
    fig, ax = plt.subplots()
    
    # print(mus_all.shape)
    y = mus_all[:,:,:,60:89].mean(axis=1)
    # y = mus_all.mean(axis=1)

    x = np.arange(0, y.shape[1], 1)/config['fit_per_epoch']
    ci = mus_all.std(axis=1)*2

    legend = []
    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']
    algs = config['alg']
    # markers = ['.', '.',  '.',  '.']
    for j in range(y.shape[0]):
        # for i in range(y.shape[2]):
        y_mean = y.mean(axis=-1, keepdims=True)
        for i in range(1):
            # ax.plot(x,y[j,:,i], linestyle=linestyles[j], color=colors[i])
            # ax.scatter(x,y[j,:,i], linestyle=linestyles[j], color=colors[i], marker='.')
            ax.scatter(x,y_mean[j,:,i], color=colors[j+1], s=5)
            legend += [str(algs[j])+str(i)]
        # ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

    ax.set_xlabel('epochs')
    ax.set_ylabel('average policy mean difference')

    plt.title(f'Accumulated Parameters {i}')
    plt.legend(legend)
    plt.savefig(f'imgs/{exp_name}/{exp_name}_mus_diff.png')

    if y.shape[-1] > 3:
        return

    legend = []
    for i in range(y.shape[-1]):
    
        fig, ax = plt.subplots()

        for j in range(y.shape[0]):
            # ax.plot(x,y[j,:,i], linestyle=linestyles[j], color=colors[i])
            # ax.scatter(x,y[j,:,i], linestyle=linestyles[j], color=colors[i], marker='.')
            ax.scatter(x,y[j,:,i], color=colors[j], marker=markers[j], s=5)
            legend += [str(algs[j])+str(i)]
        # ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

        ax.set_xlabel('epochs')
        ax.set_ylabel('average policy mean difference')

        plt.title(f'Parameter {i}')
        plt.legend(legend)
        plt.savefig(f'imgs/{exp_name}/{exp_name}_mus_diff_{i}.png')

def plot_kl(kls_all, exp_name, legend, config):

    kls_all = np.array(kls_all)
    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
    
    fig, ax = plt.subplots()
    y = kls_all.mean(axis=1)
    x = np.arange(0, y.shape[1], 1)/config['fit_per_epoch']
    ci = kls_all.std(axis=1)*2

    # REMOVE
    for j in range(y.shape[0]):
        y[j][y[j]<-.5] = 0
        ax.plot(x,y[j])

        # ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

    plt.hlines(config['eps'], 0, len(x), 'red', label='KL bound')
    legend += ['KL bound']

    ax.set_xlabel('epochs')
    ax.set_ylabel('average KL')

    plt.title(f'Average KL for lqr_dim {init_params["lqr_dim"]} n_ineff {init_params["n_ineff"]}')
    plt.legend(legend)
    plt.savefig(f'imgs/{exp_name}/{exp_name}_kls.png')

if __name__ == '__main__':

    exp_name = 'lqr_dim_10_eff_3_env_0_con_sample_all'

    data_dir = os.path.join('logs', exp_name)
    returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, mus_all, kls_all, legend, init_params = load_data_from_dir(data_dir)
    
    print('alg', legend)
    # print('max min', np.array(returns_mean_all).max(), np.array(returns_mean_all).min())
    print('optimal_reward', np.array(optimal_reward_all).mean(axis=1))
    print('best_reward', np.array(best_reward_all).mean(axis=1))
    
    # exp_name = 'segway_2_clearer'
    # data_dir = os.path.join('logs', exp_name)
    # returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, legend, init_params = load_data_from_dir_segway(data_dir)
    
    plot_data(returns_mean_all, returns_std_all, optimal_reward_all, legend, exp_name, init_params)
    plot_mi(mi_avg_all, 1, exp_name, init_params)

    plot_mu(mus_all, exp_name, init_params)
    plot_mu_diff(mus_all, exp_name, init_params)
    plot_kl(kls_all, exp_name, legend, init_params)

    # for dim in [3, 5, 7]:
    #     exp_name = f'lqr_{dim}'
    #     data_dir = os.path.join('logs', exp_name)

    #     returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, legend, init_params= load_data_from_dir(data_dir)
    #     plot_data(returns_mean_all, returns_std_all, optimal_reward_all, legend, exp_name, init_params)
    #     plot_mi(mi_avg_all, 1, exp_name, init_params)

    #     exp_name = f'lqr_{dim}_env_0'
    #     data_dir = os.path.join('logs', exp_name)

    #     returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, legend, init_params= load_data_from_dir(data_dir)
    #     plot_data(returns_mean_all, returns_std_all, optimal_reward_all, legend, exp_name, init_params)
    #     plot_mi(mi_avg_all, 1, exp_name, init_params)