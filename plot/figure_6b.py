import os
import utils

if __name__ == '__main__':

    exp_name = 'CR_AirHockey'
    title = 'AirHockey - Full Cov'
    
    data_path = os.path.join('/work/scratch/pl29zovi/', exp_name)
    out_path = '/work/scratch/pl29zovi/DR-CREPS'

    data_dict = utils.load_data_from_dir(data_path)

    labels = ['TRPO', 'PPO', 'NES', 'MORE', 'DR-CREPS (PCC)', 'DR-CREPS (MI)', 'CREPS']
    colors = ['red', 'orange', 'yellow', 'teal', 'blue', 'magenta', 'cyan']
    line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']
    
    x_lim = 10000
    y_0, y_1 = 0, 150
    x_0, x_1 = 0, x_lim
    x_ticks = 2500
    y_ticks = 50

    utils.plot_data(data_dict, exp_name, title,
                labels, colors, line_styles,
                x_lim=x_lim, max_runs=20, 
                axis=[y_0, y_1, y_ticks, x_0, x_1, x_ticks],
                optimal_key=None,
                smooth_algo=['Nelder-Mead simplex'],
                filename='results_airhockey', out_path=out_path, pdf=True)