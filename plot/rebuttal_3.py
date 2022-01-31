import os
import utils

if __name__ == '__main__':

    exp_name = 'CR_BallStopping_apx'
    title = ''
    
    data_path = os.path.join('/work/scratch/pl29zovi/', exp_name)
    out_path = '/work/scratch/pl29zovi/DR-CREPS'

    data_dict = utils.load_data_from_dir(data_path)

    labels = ['TRPO', 'REINFORCE', 'PPO', 'Nelder-Mead simplex', 'NES', 'L-BFGS', 'ES', 'DR-CREPS (PCC)', 'CEM']
    colors = ['red', 'black', 'orange', 'grey', 'yellow', 'teal', 'cyan', 'blue', 'magenta']
    line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

    x_lim = 7000
    y_0, y_1 = -35, 10
    x_0, x_1 = 0, x_lim
    x_ticks = 2000
    y_ticks = 10
    
    utils.plot_data(data_dict, exp_name, title,
                labels, colors, line_styles,
                x_lim=x_lim, max_runs=23, 
                axis=[y_0, y_1, y_ticks, x_0, x_1, x_ticks],
                optimal_key=None,
                filename='results_ballstopping_apx', out_path=out_path, pdf=True)