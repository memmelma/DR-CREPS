import joblib
from ball_mi_el import experiment

if __name__ == '__main__':
    
    exp_name = 'ball'
    load_exp = 'ConstrainedREPSMI_0'

    params = joblib.load(f'logs/{exp_name}/{load_exp}')['init_params']
    state = joblib.load(f'logs/{exp_name}/{load_exp}_state')['distribution']

    params['sigma_init'] = state
    params['quiet'] = False

    experiment(**params)

    