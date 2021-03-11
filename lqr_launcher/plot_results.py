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
    legend = []
    optimal_reward_all = []
    best_reward_all = []

    subdirs = os.listdir(run)
    for subdir in subdirs:
        if subdir[-4:] != '.txt' and subdir[-3:] != '.sh':
            legend += [subdir]
            files = os.listdir(os.path.join(run, subdir))
            returns_mean = None
            returns_std = None
            gain_lqr = None
            gain_policy = None
            init_params = None
            mi_avg = None
            optimal_reward = None
            best_reward = None

            file_count = 0
            print(os.path.join(run, subdir))
            for file in files:
                
                if file[-4:] != '.txt':
                    data = joblib.load(os.path.join(run, subdir, file))

                    # dict_keys(['returns_mean', 'returns_std', 'agent', 'gain_lqr', 'gain_policy', 'optimal_reward', 'best_reward', init_params', 'alg', 'mi_avg'])
                    
                    # print(f"{data['alg'].__name__} - optimal: {data['optimal_reward']} - best: {data['best_reward']})")

                    if returns_mean is None:
                        print(data['alg'].__name__, data.keys())
                        print(data['ineff_params'])
                        returns_mean =[data['returns_mean']]
                        returns_std = [data['returns_std']]
                        gain_lqr = [data['gain_lqr']]
                        gain_policy = [data['gain_policy']]
                        optimal_reward = [data['optimal_reward']]
                        best_reward = [data['best_reward']]

                        init_params = data['init_params']
                        
                        if data['alg'].__name__ == 'REPS_MI' or data['alg'].__name__ == 'REPS_MI_CON':
                            mi_avg = [data['mi_avg']]

                    else:
                        returns_mean += [data['returns_mean']]
                        returns_std += [data['returns_std']]

                        gain_lqr += [data['gain_lqr']]
                        gain_policy += [data['gain_policy']]
                        
                        optimal_reward += [data['optimal_reward']]
                        best_reward += [data['best_reward']]

                        if data['alg'].__name__ == 'REPS_MI' or data['alg'].__name__ == 'REPS_MI_CON':
                            mi_avg += [data['mi_avg']]
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
            optimal_reward_all += [optimal_reward]
            best_reward_all += [best_reward]

    return returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, legend, init_params

# def plot_data(returns_mean_all, returns_std_all, legend, exp_name, config):

#     os.makedirs('imgs', exist_ok=True)

#     fig, ax = plt.subplots()

#     y = np.array(returns_mean_all).mean(axis=1)
#     # print(y.shape)
#     x = np.arange(0, y.shape[1], 1)
#     ci = np.array(returns_mean_all).std(axis=1)
#     # np.array(returns_std_all)

#     for i in range(y.shape[0]):
#         ax.plot(x,y[i])
#         ax.fill_between(x, (y-ci)[i], (y+ci)[i], alpha=.3)
    
#     plt.title(f'samples {config["ep_per_fit"]} w/ {legend}')
#     # plt.legend([alg.__name__ for alg in algs])
#     plt.legend(legend)
#     plt.savefig(f'imgs/samples_{config["ep_per_fit"]}_dim_{config["lqr_dim"]}_{exp_name}.png')

def plot_data(returns_mean_all, returns_std_all, optimal_reward_all, legend, exp_name, config):

    os.makedirs('imgs', exist_ok=True)

    fig, ax = plt.subplots()
    
    for i in range(len(returns_mean_all)):
        y = np.array(returns_mean_all[i]).mean(axis=0)

        # print(y.shape)
        x = np.arange(0, y.shape[0], 1)
        ci = np.array(returns_mean_all[i]).std(axis=0)

        # print(np.array(returns_mean_all[i])[:,0])
        # print(np.array(returns_mean_all[i])[:,0].shape)
        # print(np.array(returns_mean_all[i])[:,0].mean(), np.array(returns_mean_all[i])[:,0].std())

        ax.plot(x,y, label=legend[i])
        ax.fill_between(x, (y-ci), (y+ci), alpha=.3)
        # ax.errorbar(x, y, yerr=ci, errorevery=10)

    ax.set_ylim(-500,0)
    # ax.set_xlim(0,50)
    
    ax.set_xlabel('epochs')
    ax.set_ylabel('total return')
    
    plt.hlines(np.array(optimal_reward_all[0]).mean(), 0, len(x), 'red', label='optimal control')

    plt.legend()
    plt.savefig(f'imgs/{exp_name}.pdf')

    plt.title(f'samples {config["ep_per_fit"]} w/ {legend}')
    plt.savefig(f'imgs/{exp_name}.png')

def plot_mi(avg_mi_all, mi_idx, exp_name, config):

    avg_mi_all = np.array(avg_mi_all[mi_idx])
    os.makedirs('imgs', exist_ok=True)
    
    fig, ax = plt.subplots()

    y = avg_mi_all.mean(axis=0)
    # y = np.array(avg_mi_all).mean(axis=0)

    x = np.arange(0, y.shape[0], 1)/config['fit_per_epoch']
    # ci = np.array(avg_mi_all).std(axis=0)
    ci = avg_mi_all.std(axis=0)

    for i in range(y.shape[1]):
        ax.plot(x,y[:,i])
        ax.fill_between(x, (y-ci)[:,i], (y+ci)[:,i], alpha=.3)

    ax.set_xlabel('epochs')
    ax.set_ylabel('average mutual information')

    plt.savefig(f'imgs/{exp_name}_mi.pdf')

    plt.title(f'MI samples {config["ep_per_fit"]} w/ {legend[mi_idx]}')
    # plt.legend(list(range(y.shape[1])))
    plt.savefig(f'imgs/{exp_name}_mi.png')

if __name__ == '__main__':

    # exp_name = 'lqr_3_env_0'
    # exp_name = 'lqr_7'
    # data_dir = os.path.join('logs', exp_name)

    # returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, legend, init_params= load_data_from_dir(data_dir)
    
    # print(best_reward_all)
    # print('alg', legend)
    # print('best_reward', np.array(best_reward_all).mean(axis=1))
    # print('optimal_reward', np.array(optimal_reward_all).mean(axis=1))
    # print('max min', np.array(returns_mean_all).max(), np.array(returns_mean_all).min())
    
    # plot_data(returns_mean_all, returns_std_all, optimal_reward_all, legend, exp_name, init_params)
    # plot_mi(mi_avg_all, 1, exp_name, init_params)


    for dim in [3, 5, 7]:
        exp_name = f'lqr_{dim}'
        data_dir = os.path.join('logs', exp_name)

        returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, legend, init_params= load_data_from_dir(data_dir)
        plot_data(returns_mean_all, returns_std_all, optimal_reward_all, legend, exp_name, init_params)
        plot_mi(mi_avg_all, 1, exp_name, init_params)

        exp_name = f'lqr_{dim}_env_0'
        data_dir = os.path.join('logs', exp_name)

        returns_mean_all, returns_std_all, optimal_reward_all, best_reward_all, mi_avg_all, legend, init_params= load_data_from_dir(data_dir)
        plot_data(returns_mean_all, returns_std_all, optimal_reward_all, legend, exp_name, init_params)
        plot_mi(mi_avg_all, 1, exp_name, init_params)