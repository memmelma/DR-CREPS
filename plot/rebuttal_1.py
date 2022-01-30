import os
import utils

if __name__ == '__main__':

    exp_name = 'ship_rebuttal'
    title = ''
    
    data_path = os.path.join('/work/scratch/pl29zovi/', exp_name)
    out_path = '/work/scratch/pl29zovi/DR-CREPS'

    data_dict = utils.load_data_from_dir(data_path)

    labels = ['TRPO', 'REINFORCE', 'PPO', 'Nelder-Mead simplex', 'NES', 'L-BFGS', 'ES', 'DR-CREPS (PCC)', 'CEM']
    colors = ['red', 'black', 'orange', 'grey', 'yellow', 'teal', 'cyan', 'blue', 'magenta']
    line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

    x_lim = 3500
    y_0, y_1 = -100, -55
    x_0, x_1 = 0, x_lim
    x_ticks = 1000
    y_ticks = 10
    
    utils.plot_data(data_dict, exp_name, title,
                labels, colors, line_styles,
                x_lim=x_lim, max_runs=24, 
                axis=[y_0, y_1, y_ticks, x_0, x_1, x_ticks],
                # legend_params={'loc':'upper left', 'bbox_to_anchor': (-0.15,-0.3), 'fontsize': 12, 'prop': {'size': 12}, 'ncol': 10},
                # legend_params={'loc':'lower right', 'fontsize': 12, 'prop': {'size': 12}, 'ncol': 1},
                optimal_key='optimal reward',
                filename='results_shipsteering_apx', out_path=out_path, pdf=True)