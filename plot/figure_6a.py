import os
import utils
from matplotlib.lines import Line2D

if __name__ == '__main__':

    exp_name = 'CR_ShipSteering'
    title = 'ShipSteering - Full Cov'
    
    data_path = os.path.join('/work/scratch/pl29zovi/', exp_name)
    out_path = '/work/scratch/pl29zovi/DR-CREPS'

    data_dict = utils.load_data_from_dir(data_path)

    labels = ['TRPO', 'PPO', 'NES', 'MORE', 'DR-REPS (PCC)', 'DR-REPS (MI)', 'DR-CREPS (PCC)', 'DR-CREPS (MI)', 'CREPS']
    colors = ['red', 'orange', 'yellow', 'teal', 'grey', 'black', 'blue', 'magenta', 'cyan']
    line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']
    
    x_lim = 3500
    y_0, y_1 = -100, -55
    x_0, x_1 = 0, x_lim
    x_ticks = 1000
    y_ticks = 10
    
    legend_elements = [Line2D([0], [0], color='#BBBBBB', lw=1, label='DR-REPS (PCC)'), 
                        Line2D([0], [0], color='#000000', lw=1, label='DR-REPS (MI)')]
    
    utils.plot_data(data_dict, exp_name, title,
                labels, colors, line_styles,
                x_lim=x_lim, max_runs=2, 
                axis=[y_0, y_1, y_ticks, x_0, x_1, x_ticks],
                legend_params={'handles': legend_elements, 'loc': 'lower right', 'fontsize': 12, 'prop': {'size': 12}, 'ncol': 1},
                # legend_params={'loc':'upper left', 'bbox_to_anchor': (-0.15,-0.3), 'fontsize': 12, 'prop': {'size': 12}, 'ncol': 10},
                optimal_key='optimal reward',
                out_path=out_path, pdf=False)

