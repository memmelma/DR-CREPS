import joblib
import os

from experiments.ball_stopping.el_ball_stopping import experiment

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--algo', type=int, nargs='+')
    parser.add_argument('--seed', type=int, nargs='+')

    args = parser.parse_args()
    print(args, args.algo)
    # for exp_name, load_exp in zip(
    #             ['ball_full_fix/alg_ConstrainedREPS/eps_4.5/kappa_20.0/distribution_cholesky/sample_type_None/ep_per_fit_250/n_epochs_28',
    #             'ball_full_fix/alg_ConstrainedREPSMIFull/eps_4.5/kappa_20.0/distribution_mi/sample_type_percentage/method_MI/gamma_0.5/k_30/ep_per_fit_60/n_epochs_116',
    #             'ball_full_fix/alg_MORE/eps_4.5/kappa_20.0/distribution_cholesky/sample_type_None/ep_per_fit_250/n_epochs_28'],
                
    #             ['ConstrainedREPS', 
    #             'ConstrainedREPSMIFull', 
    #             'MORE']):

    # for i in range(10):
    
    exp_names = ['ball_full_fix/alg_ConstrainedREPS/eps_4.5/kappa_20.0/distribution_cholesky/sample_type_None/ep_per_fit_250/n_epochs_28',
                'ball_full_fix/alg_ConstrainedREPSMIFull/eps_4.5/kappa_20.0/distribution_mi/sample_type_percentage/method_MI/gamma_0.5/k_30/ep_per_fit_60/n_epochs_116',
                'ball_full_fix/alg_MORE/eps_4.5/kappa_20.0/distribution_cholesky/sample_type_None/ep_per_fit_250/n_epochs_28']
    
    exp_name = exp_names[args.algo[0]]
    load_exps = ['ConstrainedREPS', 
                'ConstrainedREPSMIFull', 
                'MORE']

    load_exp = load_exps[args.algo[0]]
    i = args.seed[0]

    exp = load_exp+f'_{i}'

    params = joblib.load(f'logs/{exp_name}/{exp}')['init_params']
    state = joblib.load(f'logs/{exp_name}/{exp}_state')['distribution']

    params['sigma_init'] = state
    params['quiet'] = False
    params['save_render_path'] = os.path.join('video_log', 'ball_stopping', exp)
    params['ep_per_fit'] = 5
    params['n_epochs'] = 1

    print(params)
    experiment(**params)
