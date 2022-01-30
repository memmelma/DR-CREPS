import os
import joblib
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import distributions
# plt.style.use('tableau-colorblind10')

# Code taken from https://github.com/MushroomRL/mushroom-rl-benchmark/blob/28872e5d09e9afabba0ece8cbf827d296d427af4/mushroom_rl_benchmark/utils/plot.py
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
    assert bool(data_dict_all), f'No data found at {data_dir}!'
    return data_dict_all

def plot_data(data_dict, exp_name, title, labels, colors, line_styles, x_lim=5000, max_runs=25, axis=None, legend_params=None, optimal_key=None, out_path='', smooth_algo=[], pdf=False):

    # Color blind safe colors
    # https://github.com/garrettj403/SciencePlots/blob/ed7efb91447c33ce0cc05df1c6b28efbd14e5589/styles/color/vibrant.mplstyle
    # https://personal.sron.nl/~pault/#fig:scheme_bright
    # combination of 'Vibrant' colours extended by 'yellow' ('Bright') and 'black' ('High-contrast') 
    color_dict = dict({'orange': '#EE7733', 'blue': '#0077BB', 'cyan': '#33BBEE', 'magenta': '#EE3377', 'red': '#CC3311', 'teal': '#009988', 'grey':'#BBBBBB', 'yellow': '#CCBB44', 'black': '#000000'})
        
    out_path = os.path.join(out_path, f'imgs/{exp_name}')
    os.makedirs(out_path, exist_ok=True)
    fig, ax = plt.subplots()

    for i_exp, exp in enumerate(sorted(data_dict.keys(), reverse=True)):

        assert len(data_dict[exp]['returns_mean']) >= max_runs, f"Some runs of {exp} might have failed! runs {len(data_dict[exp]['returns_mean'])} < max_runs {max_runs}"

        exp_dict = data_dict[exp]
        init_params = exp_dict['init_params'][0]

        cut = (x_lim // init_params['ep_per_fit']) + 1
        max_len = max(np.max([len(x) for x in exp_dict['returns_mean']]), cut)

        def make_same_length(x, max_len):
            if len(x) < max_len:
                x += list(np.ones(max_len - len(x))*x[-1])
            return x[:max_len]
        
        y, ci  = get_mean_and_confidence([make_same_length(x, max_len) for x in exp_dict['returns_mean']])
        
        y = y[:cut]
        ci = ci[1][:cut]

        if labels[i_exp] in smooth_algo:
            def smooth(y, box_pts):
                box = np.ones(box_pts)/box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth
            y = smooth(y,50)
            ci = smooth(ci,50)

        x = np.arange(0, cut, 1) * init_params['ep_per_fit']

        ax.plot(x, y, label=labels[i_exp], color=color_dict[colors[i_exp]], ls=line_styles[i_exp], linewidth=2)
        ax.fill_between(x, (y-ci), (y+ci), color=color_dict[colors[i_exp]], alpha=.3)
            
        params = exp.split('|')[2:]
        print(init_params['alg'], 'max J:', np.round(np.max(y), 4), sorted(params))
        print('Completed runs', len(exp_dict['returns_mean']))

    if optimal_key is not None:
        plt.hlines(np.array(exp_dict['optimal_reward'][0]).mean(), 0, x.max(), 'red', label=optimal_key, ls='dashed')
    
    if axis is not None:
        y_0, y_1 = axis[0], axis[1]
        x_0, x_1 = axis[3], min(x_lim, axis[4])
        y_ticks, x_ticks = axis[2], axis[5]
        plt.xticks(range(x_0, x_1+x_ticks, x_ticks), fontsize=20)
        plt.yticks(range(y_0, y_1+y_ticks, y_ticks), fontsize=20)
        plt.ylim(y_0, y_1)
        plt.xlim(x_0, x_1)

    ax.set_xlabel('episodes', fontsize=20)
    ax.set_ylabel(r'$J$', fontsize=20)

    if legend_params is not None:
        plt.legend(**legend_params)

    plt.title(title, fontsize=20)

    plt.tight_layout()
    plt.grid()
    
    plt.savefig(os.path.join(out_path, f"returns.{'pdf' if pdf else 'png'}"), bbox_inches='tight', pad_inches=0)
    plt.close()