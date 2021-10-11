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

        # if data_dict[exp]['init_params'][0]['sample_type'] != None and data_dict[exp]['init_params'][0]['gamma'] != 0.5:
        #     continue

        # if data_dict[exp]['init_params'][0]['alg'] != 'REPS' and data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPS':
        #     continue

        ####
        # if data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPS' and data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPSMIFull':
        #     continue
        # if data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPS' and data_dict[exp]['init_params'][0]['gamma'] != 0.1:
        #     continue

        # if data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPS' and data_dict[exp]['init_params'][0]['k'] != 50:
        #     continue
        
        # if data_dict[exp]['init_params'][0]['ep_per_fit'] != 15 and data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPS':
        #     continue
            
        # if data_dict[exp]['init_params'][0]['ep_per_fit'] != 50 and data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPSMIFull':
        #     continue


        # if data_dict[exp]['init_params'][0]['sample_type'] != 'percentage':
        #     continue
        
        # if data_dict[exp]['init_params'][0]['gamma'] != 0.1:
        #     continue
        
        # if data_dict[exp]['init_params'][0]['k'] != 30:
        #     continue
        
        if data_dict[exp]['init_params'][0]['gamma'] != 0.1:
            continue

        # if data_dict[exp]['init_params'][0]['k'] != 200:
        #     continue


        # if data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPSMIFull' and data_dict[exp]['init_params'][0]['ep_per_fit'] > 150:
        #     continue

        ####

        # if data_dict[exp]['init_params'][0]['alg'] != 'ConstrainedREPS' and data_dict[exp]['init_params'][0]['sample_type'] != 'None':
        #     continue
        # if data_dict[exp]['init_params'][0]['alg'] != 'REPS' and data_dict[exp]['init_params'][0]['ep_per_fit'] != 150:
        #     continue
        # if data_dict[exp]['init_params'][0]['alg'] != 'REPS' and data_dict[exp]['init_params'][0]['gamma'] != 0.5:
        #     continue
        # if data_dict[exp]['init_params'][0]['alg'] != 'REPS' and data_dict[exp]['init_params'][0]['k'] != 50:
        #     continue
        # if data_dict[exp]['init_params'][0]['eps'] != 0.7:
        #     continue

        # if data_dict[exp]['init_params'][0]['distribution'] != 'cholesky' and data_dict[exp]['init_params'][0]['distribution'] != 'mi':
        #     continue

        # if data_dict[exp]['init_params'][0]['method'] != 'MI':
        #     continue

        # if data_dict[exp]['init_params'][0]['ep_per_fit'] != 125 and data_dict[exp]['init_params'][0]['ep_per_fit'] != 250:
        #     continue
        
        # if data_dict[exp]['init_params'][0]['alg'] == 'ConstrainedREPSMIFull':
        #     if data_dict[exp]['init_params'][0]['sample_type'] == 'percentage':
        #         if data_dict[exp]['init_params'][0]['gamma'] != 0.1:
        #             continue

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
        if x_axis == 'samples':
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
                labels = ['RWR', 'PRO', 'RWR w/ PE', 'REPS', 'REPS w/ PE', 'CREPS', 'CREPS w/ PE']
                # labels = ['RWR', r'PRO $\beta=0.2$', 'RWR w/ PE', 'REPS $\epsilon=0.4$', 'REPS w/ PE $\epsilon=0.4$', 'CREPS $\epsilon=2.5, \kappa=6.0$', 'CREPS w/ PE $\epsilon=2.5, \kappa=6.0$']
                colors = ['m', 'tab:olive', 'tab:purple', 'tab:blue', 'tab:blue', 'tab:orange', 'tab:orange']
                line_styles = ['solid', 'solid', 'dashed', 'dashed', 'solid', 'dashed', 'solid']
                
            elif exp_name == 'lqr_full/alg_plot':
                labels = ['RWR', 'REPS', 'REPS w/ PE (ours)', 'CREPS', 'constrained REPS-PE (ours)']
                
            elif exp_name == 'lqr_full/alg_ablation_mi_pearson':
                labels = ['MORE', 'CREPS w/ diag. cov.', 'CREPS w/ full cov.', 'DR-CREPS (PCC)', 'DR-CREPS w/o PE (PCC)', 'DR-CREPS (MI)', 'DR-CREPS w/o PE (MI)']
                colors = ['tab:green', 'tab:orange', 'tab:orange', 'tab:pink', 'tab:pink', 'tab:purple', 'tab:purple']
                line_styles = ['solid', 'dashed', 'solid', 'solid', 'dashed', 'solid', 'dashed']
                

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

            ax.plot(x*n_samples,y, label=label, color=color, ls=ls, linewidth=3 if 'pink' in color else 2)
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

    if 'bullet' in exp:
        plt.ylim(0, 250)
        plt.ylim(-200, 250)
    
    # plt.xlim(0, 5000)

    if not clean:
        plt.title(exp_name)
    plt.tight_layout()
    plt.grid()
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

    fig, ax = plt.subplots()

    vio_ctr = 0
    tick_legend = []

    x_width = 0.5
    mean_width = 3.

    for i_exp, exp in enumerate(sorted(data_dict.keys(), reverse=True)):

        init_params = data_dict[exp]['init_params'][0]
        a = init_params['oracle']
        b = data_dict[exp]['top_k_mis']

        tag = 'None' # 'percentage'
        if init_params['sample_type'] != tag:
            continue
        if 'Oracle' in init_params['alg']:
            continue

        b_new = []
        for b_i in b:
            b_tmp = []
            for b_j in b_i:
                b_tmp += [len(np.intersect1d(a,b_j).tolist())]
            b_new += [b_tmp]
        
        b_new = np.array(b_new) / len(init_params['oracle'])

        y = np.mean(b_new, axis=1)
        violin = ax.violinplot(y, [x_width*vio_ctr], 
                                showmeans=False, showmedians=False, showextrema=True)#, quantiles=[0.25, 0.75])

        color = 'tab:red' if 'Oracle' in init_params['alg'] \
                    else 'tab:blue' if init_params['method'] == 'MI' \
                    else 'tab:orange' if init_params['method'] == 'Pearson' \
                    else 'green'
                    
        # ax.scatter([x_width*vio_ctr], np.median(y), marker='o',color='black', s=30, zorder=3)
        # ax.scatter([x_width*vio_ctr], np.median(y), marker='o',color='white', s=10, zorder=3)
        ax.scatter([x_width*vio_ctr], np.median(y), marker='o', color='white', s=5, zorder=3)

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
        tick_legend += ['ORACLE' if 'Oracle' in init_params['alg'] else f"{init_params['k']}"]

    plt.ylabel('percentage of correct parameters')
    plt.xlabel('top $m$ selected parameters')

    plt.xticks(np.arange(0, len(tick_legend), 1.)*x_width, tick_legend)
    
    plt.ylim(0.2,0.9)

    custom_lines = [Line2D([0], [0], color='tab:orange', lw=mean_width),
                    Line2D([0], [0], color='tab:blue', lw=mean_width)]
    ax.legend(custom_lines, ['PCC', 'MI'], loc='lower left')

    plt.tight_layout()
    plt.savefig(f"imgs/{exp_name}/parameters_sample_type_{tag}.{'pdf' if pdf else 'png'}")
    plt.close()

if __name__ == '__main__':

    pdf = False

    exp_name = 'ship_3_tiles_full_mi_vs_random_FULL'
    exp_name = 'hockey_mix_eps'
    exp_name = 'lqr_mi_store'
    exp_name = 'hockey_eps_kappa/alg_RWR'
    exp_name = 'lqr_diag/alg_Best'#REPS_MI'
    exp_name = 'hockey_bound_eps_kappa/alg_ConstrainedREPS'
    exp_name = 'lqr_full/alg_ConstrainedREPSMIFull'
    # exp_name = 'lqr_full/alg_ConstrainedREPS'
    # exp_name = 'lqr_full/alg_best'
    # exp_name = 'ship/3_tiles_best'
    # exp_name = 'ship_diag/alg_ConstrainedREPSMI'
    # exp_name = 'lqr_diag/alg_RWR_MI'#REPS_MI'
    # exp_name = 'ship_3_tiles_full_besty_10/alg_REPS_MI_full/'
    exp_name = 'lqr_diag/alg_plot'
    exp_name = 'lqr_full/alg_plot'
    # exp_name = 'lqr_full/alg_REPS_MI_full'
    max_runs = 25
    exp_name = 'hockey_bound_eps_kappa/alg_REPS'
    exp_name = 'hockey_bound_gamma_k/alg_ConstrainedREPSMIFull'
    # exp_name = 'hockey_bound_eps_kappa/alg_ConstrainedREPS'
    # exp_name = 'hockey_sanity'
    exp_name = 'hockey_bound_gamma_k/alg_REPS_MI_full'
    # exp_name = 'hockey_bound_gamma_k/alg_MORE'
    # exp_name = 'hockey_bound_eps_kappa/alg_ConstrainedREPS'
    exp_name = 'lqr_fix'
    exp_name = 'ship_fix'
    exp_name = 'lqr_full/alg_MI_vs_Pearson'
    exp_name = 'lqr_diag/alg_plot'
    # exp_name = 'lqr_full/alg_ConstrainedREPSMIFull/distribution_mi/eps_4.7/kappa_17.0/k_50'
    # exp_name = 'lqr_full/ablation_full'
    exp_name = 'lqr_diag/alg_plot'
    exp_name = 'ship/alg_plot'
    # exp_name = 'ship_3_tiles_full_mi_vs_random_FULL'
    max_runs = 8
    # exp_name = 'lqr_full/alg_ablation_mi_pearson'
    # max_runs = 10

    exp_name = 'ship_fix/alg_plot_02'
    exp_name = 'lqr_full/alg_ablation_mi_pearson'
    exp_name = 'lqr_diag/alg_plot'
    # exp_name = 'hockey_full'
    # exp_name = 'lqr_full/alg_ablation_mi_pearson'
    exp_name = 'ball_full_fix'
    max_runs = 25
    # exp_name = 'hockey_full'
    # exp_name = 'hockey_bound_fix/alg_REPS_MI_full'
    # exp_name = 'hockey_mix_eps_fix/alg_ConstrainedREPSMIFull/eps_2.2/kappa_15.0/k_30/sample_type_percentage'
    # max_runs = 5

    exp_name = 'lqr_compare_params_oracle/alg_REPS_MI/distribution_diag/eps_0.7/k_50'
    exp_name = 'lqr_ablation'
    max_runs = 1
    # exp_name = 'lqr_ablation_k_gamma'
    # exp_name = 'lqr_ablation_k_gamma_full'
    # max_runs = 25

    data_dir = os.path.join('logs', exp_name)
    data_dict = load_data_from_dir(data_dir)

    plot_data(data_dict, exp_name, episodes=1000, samples=-1, x_axis='samples', pdf=pdf, max_runs=max_runs, clean=False)

    # plot_data(data_dict, exp_name, episodes=1000, samples=-1, x_axis='samples', pdf=True, max_runs=max_runs, clean=True)
    # plot_data(data_dict, exp_name, episodes=1000, samples=-1, x_axis='samples', pdf=False, max_runs=max_runs, clean=True)

    plot_parameter(data_dict, exp_name)

    # # plot_mi(data_dict, exp_name, pdf=pdf)
    # plot_kl(data_dict, exp_name, pdf=pdf)
    # plot_entropy(data_dict, exp_name, pdf=pdf)
    