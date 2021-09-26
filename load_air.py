import joblib
from el_air_hockey_mi import experiment

if __name__ == '__main__':
    
    exp_name = 'hockey_eps_kappa/alg_ConstrainedREPS/eps_3.2/kappa_2.0'
    load_exp = 'ConstrainedREPS_0'

    params = joblib.load(f'logs/{exp_name}/{load_exp}')['init_params']
    state = joblib.load(f'logs/{exp_name}/{load_exp}_state')['distribution']

    params['sigma_init'] = state
    params['ep_per_fit'] = 1
    params['quiet'] = False
    
    print(params)
    experiment(**params)
