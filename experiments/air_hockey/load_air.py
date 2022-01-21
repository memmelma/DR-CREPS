import joblib
from experiments.air_hockey.el_air_hockey import experiment
import os
if __name__ == '__main__':
    
    for exp_name, load_exp in zip(
                ['hockey_full/alg_ConstrainedREPS/eps_2.0/kappa_12.0/distribution_cholesky/ep_per_fit_250/n_epochs_40', 
                'hockey_full/alg_ConstrainedREPSMIFull/eps_2.0/kappa_12.0/k_30/sample_type_percentage/gamma_0.5/method_MI/distribution_mi/ep_per_fit_50/n_epochs_200', 
                'hockey_full/alg_MORE/eps_2.4/kappa_12.0/distribution_cholesky/ep_per_fit_250/n_epochs_40'],
                ['ConstrainedREPS', 
                'ConstrainedREPSMIFull', 
                'MORE']):

        for i in range(10):

            exp = load_exp+f'_{i}'
            params = joblib.load(f'logs/{exp_name}/{exp}')['init_params']
            state = joblib.load(f'logs/{exp_name}/{exp}_state')['distribution']

            params['sigma_init'] = state
            params['quiet'] = False
            params['save_render_path'] = os.path.join('video_log', 'air_hockey', exp)
            params['ep_per_fit'] = 5
            params['n_epochs'] = 1
            params['nn'] = 0

            print(params)
            experiment(**params)
