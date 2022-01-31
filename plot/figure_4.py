import os
import utils
from matplotlib.lines import Line2D

if __name__ == '__main__':

    exp_name = 'CR_LQR'
    title = 'LQR - Full Cov'
    
    data_path = os.path.join('/work/scratch/pl29zovi/', exp_name)
    out_path = '/work/scratch/pl29zovi/DR-CREPS'
    data_dict = utils.load_data_from_dir(data_path)

    # # CR_LQR_full
    # labels = ['TRPO', 'PPO', 'NES', 'MORE', 'DR-CREPS (PCC)', 'DR-CREPS w/o PE (PCC)', 'DR-CREPS (MI)', 'DR-CREPS w/o PE (MI)','CREPS']
    # colors = ['red', 'orange', 'yellow', 'teal', 'blue', 'blue', 'magenta', 'magenta', 'cyan']
    # line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'solid', 'dashed', 'solid']

    labels = ['TRPO', 'PPO', 'NES', 'MORE', 'DR-CREPS (PCC)', 'DR-CREPS (MI)', 'CREPS']
    colors = ['red', 'orange', 'yellow', 'teal', 'blue', 'magenta', 'cyan']
    line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

    x_lim = 5000
    y_0, y_1 = -60, 0
    x_0, x_1 = 0, x_lim
    x_ticks = 1000
    y_ticks = 10
    
    legend_elements = [Line2D([0], [0], color='red', linestyle='dashed', lw=1, label='optimal control')]

    # legend
    utils.plot_data(data_dict, exp_name, title,
                labels, colors, line_styles,
                x_lim=x_lim, max_runs=25, 
                axis=[y_0, y_1, y_ticks, x_0, x_1, x_ticks],
                legend_params={'loc':'lower right', 'bbox_to_anchor': (-1,-1), 'fontsize': 12, 'prop': {'size': 12}, 'ncol': 10}, save_legend=True,
                filename='results_lqr', out_path=out_path, pdf=True)

    # plot
    utils.plot_data(data_dict, exp_name, title,
                labels, colors, line_styles,
                x_lim=x_lim, max_runs=25, 
                axis=[y_0, y_1, y_ticks, x_0, x_1, x_ticks],
                legend_params={'handles': legend_elements, 'loc': 'lower right', 'fontsize': 16, 'prop': {'size': 16}, 'ncol': 1},
                # legend_params={'loc':'lower right', 'fontsize': 12, 'prop': {'size': 12}, 'ncol': 1},
                optimal_key='optimal control',
                filename='results_lqr', out_path=out_path, pdf=True)