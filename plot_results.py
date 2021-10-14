import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, *args, **kwargs) -> None:
        super(Network, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        n_features = 16

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('linear'))


    def forward(self, state) -> torch.Tensor:
        features1 = torch.relu(self._h1(torch.squeeze(state, 1).float()))
        a = self._h2(features1)
        a = torch.tanh(a)
        return a

def load_data_from_dir(data_dir):
    # create dict for all exps
    data_dict_all = dict()
    # walk folders
    for root, dirs, files in os.walk(data_dir):
        # if file is in folder
        if not dirs:
            # make dict for exp
            data_dict = dict()
            
            # from copy import copy
            # for i, file in enumerate(copy(files)):
            #     files[i] = os.path.join(root, file)

            # from multiprocessing import Pool
            # print(files[0])
            # func(files[0])
            # with Pool(5) as p:
            #     out = p.map(func, files)
            # print(out)

            for file in files:
                
                # if 'k_40' not in root.split('/'):
                #     continue
                
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

def plot_data(data_dict, exp_name, episodes=1000, samples=5000, x_axis='samples', n_samples = 1, max_runs=25, pdf=False, clean=False):

    os.makedirs(f'imgs/{exp_name}', exist_ok=True)
    fig, ax = plt.subplots()
    max_reward = -np.inf
    max_reward_exp = None
    max_y_discount = -np.inf
    max_y_discount_exp = None

    for i_exp, exp in enumerate(sorted(data_dict.keys(), reverse=True)):
        # CATCHES missing keys, etc.
        # catch failed experiments
        if 'init_params' not in data_dict[exp].keys():
            print(f'{exp} failed')
            continue
        if len(data_dict[exp]['returns_mean']) < max_runs:
            continue

        # for broken bullet runs
        if 'bullet' in exp:
            min_length = np.min([len(x) for x in data_dict[exp]['returns_mean']])
            data_dict[exp]['returns_mean'] = [x[:min_length] for x in data_dict[exp]['returns_mean']]
    
        try:
            y = np.array(data_dict[exp]['returns_mean']).mean(axis=0)[:episodes]
            
            # FILTERS on mean
            # if np.max(y) < 130:
            #     continue
        
        except:
            print(f'{exp} failed')
            continue
        
        # determine x axis
        if x_axis == 'episodes':
            if samples > 0:
                cut = samples // data_dict[exp]['init_params'][0]['ep_per_fit']
                x = np.arange(0, cut, 1) * ( data_dict[exp]['init_params'][0]['ep_per_fit'] // n_samples )
                y = y[:cut]
            else:
                cut = None
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

        ci = (np.array(data_dict[exp]['returns_mean']).std(axis=0)*2)[:cut]

        if clean:
            if exp_name == 'lqr_diag/alg_plot':
                labels = ['RWR', 'PRO', 'REPS', 'REPS w/ PE $\gamma=0.9$', 'REPS w/ PE $\gamma=0.1$', 'CREPS', 'CREPS w/ PE $\gamma=0.1$']
                # labels = ['RWR', r'PRO $\beta=0.2$', 'RWR w/ PE', 'REPS $\epsilon=0.4$', 'REPS w/ PE $\epsilon=0.4$', 'CREPS $\epsilon=2.5, \kappa=6.0$', 'CREPS w/ PE $\epsilon=2.5, \kappa=6.0$']
                colors = ['m', 'tab:olive', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange']
                line_styles = ['solid', 'solid', 'solid', 'dashed', 'dashdot', 'solid',  'dashed', 'solid']
                
            elif exp_name == 'lqr_full/alg_plot':
                labels = ['RWR', 'REPS', 'REPS w/ PE (ours)', 'CREPS', 'constrained REPS-PE (ours)']
                
            elif exp_name == 'lqr_full/alg_ablation_mi_pearson':
                # labels = ['MORE', 'CREPS w/ diag. cov.', 'CREPS', 'DR-CREPS (PCC)', 'DR-CREPS w/o PE (PCC)', 'DR-CREPS (MI)', 'DR-CREPS w/o PE (MI)']
                # colors = ['tab:green', 'tab:orange', 'tab:orange', 'tab:pink', 'tab:pink', 'tab:purple', 'tab:purple']
                # line_styles = ['solid', 'dashed', 'solid', 'solid', 'dashed', 'solid', 'dashed']
                labels = ['MORE', 'CREPS', 'DR-CREPS (Random)', 'DR-CREPS w/o PE (Random)', 'DR-CREPS (PCC)', 'DR-CREPS w/o PE (PCC)', 'DR-CREPS (MI)', 'DR-CREPS w/o PE (MI)']
                colors = ['tab:green', 'tab:orange', 'tab:brown', 'tab:brown', 'tab:pink', 'tab:pink', 'tab:purple', 'tab:purple']
                line_styles = ['solid', 'solid', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed']

            elif exp_name == 'ship_fix/alg_plot_02':
                labels = ['DR-REPS (PCC)', 'DR-REPS (MI)', 'MORE', 'CREPS', 'DR-CREPS (PCC)', 'DR-CREPS (MI)']
                colors = ['tab:cyan', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:purple',]
                line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
            
            elif exp_name == 'hockey_full':
                labels = ['MORE', 'CREPS', 'DR-CREPS (PCC)', 'DR-CREPS (MI)']
                colors = ['tab:green', 'tab:orange', 'tab:pink', 'tab:purple']
                line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']

            elif exp_name == 'ball_full_fix':
                labels = ['MORE', 'CREPS', 'DR-CREPS (PCC)', 'DR-CREPS (MI)']
                colors = ['tab:green', 'tab:orange', 'tab:pink', 'tab:purple']
                line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']
                
            else:
                labels = []
            label = labels[i_exp]

            handles, lbls = ax.get_legend_handles_labels()
            if 'lqr' in exp_name and 'optimal control' not in lbls:
                plt.hlines(np.array(data_dict[exp]['optimal_reward'][0]).mean(), 0, n_samples*x.max(), 'red', label='optimal control')
            elif 'ship' in exp_name and 'optimal' not in lbls:
                    plt.hlines(-58.5, 0, n_samples*x.max(), 'red', label='optimal')

            if 'RWR w/ PE' in label:
                continue
            color = colors[i_exp]
            ls = line_styles[i_exp]
            
            if 'lqr' in exp_name:
                ax.plot(x*n_samples,y, label=label, color=color, ls=ls, linewidth=3 if 'pink' in color else 2)
            else:
                ax.plot(x*n_samples,y, label=label, color=color, ls=ls, linewidth=2)
                
            ax.fill_between(x*n_samples, (y-ci), (y+ci), color=color, alpha=.3)

        else:
            label = f"{init_params['alg']} {sorted(params)}"

            ax.plot(x*n_samples,y, label=label, linewidth=2)
            ax.fill_between(x*n_samples, (y-ci), (y+ci), alpha=.3)

        # maximum reward
        if np.max(y) > max_reward:
            max_reward = np.max(y)
            max_reward_exp = exp
  
        print(data_dict[exp]['init_params'][0]['alg'], np.round(np.max(y), 4), sorted(params))

        # maximum discounted reward
        discount_factor = 0.5
        discount = np.arange(1, y.shape[0]+1, 1)**discount_factor
        discount_reward = np.sum( y * discount )
        if discount_reward > max_y_discount:
            max_y_discount = discount_reward
            max_y_discount_exp = exp

    print(f"MAX REWARD\n {data_dict[max_reward_exp]['init_params'][0]['alg']} | eps: {data_dict[max_reward_exp]['init_params'][0]['eps']} |" +
            f"kappa: {data_dict[max_reward_exp]['init_params'][0]['kappa']} | max reward: {max_reward} | k: {data_dict[max_reward_exp]['init_params'][0]['k']} | " +
            f"method: {data_dict[max_reward_exp]['init_params'][0]['method']} | sample_type: {data_dict[max_reward_exp]['init_params'][0]['sample_type']} | " +
            f"gamma {data_dict[max_reward_exp]['init_params'][0]['gamma']}" + f" | ep_per_fit {data_dict[max_reward_exp]['init_params'][0]['ep_per_fit']}" +
            f" | n_epochs {data_dict[max_reward_exp]['init_params'][0]['n_epochs']}")

    if 'optimal_reward' in data_dict[max_reward_exp].keys():
        # plt.hlines(np.array(data_dict[max_reward_exp]['optimal_reward'][0]).mean(), 0, n_samples*x.max(), 'red', label='optimal control')
        print('OPTIMAL', np.array(data_dict[max_reward_exp]['optimal_reward'][0]).mean())
    
    ax.set_xlabel(x_axis, fontsize=20)
    ax.set_ylabel('J', fontsize=20)

    if clean:
        ax.legend(fontsize=12)
    else:
        ax.legend(loc='upper left', bbox_to_anchor=(-0.2,-0.2), prop={'size': 6}, ncol=1)

    if clean:
        if 'lqr_diag' in exp:
            y_0, y_1 = -50, 0
            x_0, x_1 = 0, 2000
            x_ticks = 500
            y_ticks = 10

        elif 'lqr_full' in exp:
            y_0, y_1 = -50, 0
            x_0, x_1 = 0, 5000
            x_ticks = 1000
            y_ticks = 10

        elif 'ship' in exp:
            y_0, y_1 = -100, -55
            x_0, x_1 = 0, 3500
            x_ticks = 1000
            y_ticks = 10
        
        elif 'hockey' in exp:
            y_0, y_1 = 0, 180
            x_0, x_1 = 0, 10000
            x_ticks = 2500
            y_ticks = 50
        
        elif 'ball' in exp:
            y_0, y_1 = -30, 20
            x_0, x_1 = 0, 7000
            x_ticks = 2000
            y_ticks = 10

        plt.xticks(range(x_0, x_1+x_ticks, x_ticks), fontsize=20)
        plt.yticks(range(y_0, y_1+y_ticks, y_ticks), fontsize=20)
        plt.ylim(y_0, y_1)
        plt.xlim(x_0, x_1)

    # plt.xlim(0, 2500)

    # if 'bullet' in exp:
    #     plt.ylim(0, 250)
    #     plt.ylim(-200, 250)
    
    if 'bullet' in exp_name:
        plt.ylim(0, 250)

    if not clean:
        plt.title(exp_name)
    plt.tight_layout()
    plt.grid()
    # ratio = 10/16
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.savefig(f"imgs/{exp_name}/returns.{'pdf' if pdf else 'png'}", bbox_inches='tight', pad_inches=0)
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
                
                # if y.max() < 500:
                #     continue

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
        plt.ylim(-2000, 2500)
        plt.ylim(0, 3000)
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
    #     if 'ConstrainedREPSMIOracle' in exp:
    #         plt.clf()
    #         print(len(data_dict[exp]['kls']))
    #         print(len(data_dict[exp]['kls'][0]))
    #         print(np.array(data_dict[exp]['kls']).shape)
    #         plt.plot(np.array(data_dict[exp]['kls']).T)
    #         plt.savefig(f"imgs/{exp_name}/kl_oracle.{'pdf' if pdf else 'png'}")
    #         plt.close()
    #         return

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
        
        # if 'ConstrainedREPSMIOracle' in exp:
        #     plt.clf()
        #     plt.plot(np.array(data_dict[exp]['entropys']).T)
        #     plt.savefig(f"imgs/{exp_name}/entropy_oracle.{'pdf' if pdf else 'png'}")
        #     plt.close()
        #     return
        # else:
        #     continue
        
        # catch KL violations
        # try:
        #     if np.max(data_dict[exp]['kls']) > 10.:
        #         continue
        # except:
        #     pass

        if 'entropys' not in data_dict[exp].keys():
            continue
        if data_dict[exp]['entropys'][0] is None:
            continue
        # print(data_dict[exp]['entropys'])
        if 'bullet' in exp:
            min_length = np.min([len(x) for x in data_dict[exp]['entropys']])
            # print(min_length)
            data_dict[exp]['entropys'] = [x[:min_length] for x in data_dict[exp]['entropys']]

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

def plot_parameter(data_dir, exp_name, pdf=False):

    for measure in ['recall', 'precision']:
        fig, ax = plt.subplots()
        vio_ctr = 0
        tick_legend = []
        x_width = 0.75
        mean_width = 3.

        y_min = 1.
        y_max = 0.

        for exp in np.flip(sorted(data_dict.keys(), reverse=True)):

            init_params = data_dict[exp]['init_params'][0]
            a = init_params['oracle']
            b = data_dict[exp]['top_k_mis']

            tag = 'None'
            tag = 'percentage'
            if init_params['sample_type'] != tag:
                continue
            if 'Oracle' in init_params['alg']:
                continue
            
            # true positives
            tp = []
            for b_i in b:
                tmp = []
                for b_j in b_i:
                    tmp += [len(np.intersect1d(a,b_j).tolist())]
                tp += [tmp]
            
            if measure == 'recall':
                # tp / (tp + fn)
                arr = np.array(tp) / len(init_params['oracle'])
            elif measure == 'precision':
                # tp / (tp + fp)
                arr = np.array(tp) / init_params['k']
            else:
                continue

            y = np.mean(arr, axis=0)

            if y_min > y.min():
                y_min = y.min()
            if y_max < y.max():
                y_max = y.max()

            violin = ax.violinplot(y, [x_width*vio_ctr], 
                                    showmeans=False, showmedians=False, showextrema=True)

            color = 'tab:purple' if init_params['method'] == 'MI' \
                    else 'tab:purple' if init_params['method'] == 'MI_ALL' \
                    else 'tab:pink' if init_params['method'] == 'Pearson' \
                    else 'tab:pink' if init_params['method'] == 'PPC_ALL' \
                    else 'tab:orange' if init_params['method'] == 'Random' \
                    else 'green'

            ax.scatter([x_width*vio_ctr], np.median(y), marker='o', color='white', s=20, zorder=3)

            quant = np.percentile(y, [25,50,75])

            violin['bodies'][-1].set_color(color)
            violin['bodies'][-1].set_alpha(1.)

            color = 'black'
            violin['cbars'].set_color(color)
            violin['cmaxes'].set_color(color)
            violin['cmins'].set_color(color)
            # violin['cmeans'].set_color(color)
            # violin['cmeans'].set_linewidths(mean_width)

            ax.vlines([x_width*vio_ctr], quant[0], quant[2], color='black',linestyle='-',lw=4)

            vio_ctr = vio_ctr + 1
            if f"{init_params['k']}" not in tick_legend:
                tick_legend += [f"{init_params['k']}"]
        
        plt.xlabel('number of selected paramerers', fontsize=20)
        plt.ylabel(measure, fontsize=20)
        
        plt.xticks([x_width, 4*x_width, 7*x_width, 11*x_width], tick_legend, fontsize=15)
        tick_size = .1 if measure == 'recall' else .05
        plt.yticks(np.arange(np.round(y_min,1), np.round(y_max,1), tick_size), fontsize=20)

        custom_lines = [Line2D([0], [0], color='tab:purple', lw=mean_width), 
                        Line2D([0], [0], color='tab:pink', lw=mean_width), 
                        Line2D([0], [0], color='tab:orange', lw=mean_width)]
                        
        ax.legend(custom_lines, ['MI', 'PCC', 'Random'], loc='upper left', fontsize=15)

        plt.tight_layout()
        plt.savefig(f"imgs/{exp_name}/parameters_{measure}.{'pdf' if pdf else 'png'}")
        plt.figure(figsize=(10, 8))
        plt.close()

if __name__ == '__main__':

    pdf = False

    # exp_name = 'lqr_diag/alg_plot'
    # exp_name = 'lqr_full/alg_ablation_mi_pearson'
    # exp_name = 'ship_fix/alg_plot_02'
    # exp_name = 'hockey_full'
    # exp_name = 'ball_full_fix'
    # max_runs = 25
  
    exp_name = 'bullet_ant_fix_again'
    # exp_name = 'hockey_bound_nn_fix_again'
    max_runs = 1

    # exp_name = 'lqr_ablation_mod_AR'
    exp_name = 'lqr_ablation_fix'
    max_runs = 10

    data_dir = os.path.join('logs', exp_name)
    data_dict = load_data_from_dir(data_dir)

    # normal unclean plots
    plot_data(data_dict, exp_name, episodes=1000, samples=-1, x_axis='episodes', pdf=pdf, max_runs=max_runs, clean=False)

    # for paper plots
    # plot_data(data_dict, exp_name, episodes=1000, samples=-1, x_axis='episodes', pdf=True, max_runs=max_runs, clean=True)
    # plot_data(data_dict, exp_name, episodes=1000, samples=-1, x_axis='episodes', pdf=False, max_runs=max_runs, clean=True)

    # MI vs Pearson vs Random
    plot_parameter(data_dict, exp_name)
    plot_parameter(data_dict, exp_name, pdf=True)

    # # plot_mi(data_dict, exp_name, pdf=pdf)
    # plot_kl(data_dict, exp_name, pdf=pdf)
    # plot_entropy(data_dict, exp_name, pdf=pdf)
    