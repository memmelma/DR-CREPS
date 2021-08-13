import joblib
from el_ball_mi import experiment

if __name__ == '__main__':
    
    exp_name = 'ball_eps/alg_ConstrainedREPSMI/eps_0.7/k_25/gamma_0.1/kappa_3.0'
    exp_name = 'ball_eps/alg_ConstrainedREPS/eps_2.2/kappa_8.0'
    load_exp = 'ConstrainedREPSMI_0'
    load_exp = 'ConstrainedREPS_0'

    params = joblib.load(f'logs/{exp_name}/{load_exp}')['init_params']
    state = joblib.load(f'logs/{exp_name}/{load_exp}_state')['distribution']

    params['sigma_init'] = state
    params['quiet'] = False
    
    print(params)
    experiment(**params)
