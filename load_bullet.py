import joblib
from el_bullet_mi import experiment

if __name__ == '__main__':
    
    exp_name = 'bullet_reps_mi/alg_REPS_MI/eps_0.9/method_MI/sample_type_percentage/gamma_0.9/k_100/distribution_diag'
    load_exp = 'REPS_MI_1'

    # exp_name = 'bullet_reps_mi/alg_REPS/eps_0.005/distribution_diag'
    # load_exp = 'REPS_0'

    exp_name = 'bullet_halfcheetah/alg_REPS_MI_full/eps_0.3/method_MI/sample_type_percentage/gamma_0.3/k_300/distribution_mi'
    load_exp = 'REPS_MI_full_0'

    exp_name = 'bullet_halfcheetah/alg_REPS_MI_full/eps_0.9/method_MI/sample_type_percentage/gamma_0.3/k_300/distribution_mi'
    load_exp = 'REPS_MI_full_0'

    params = joblib.load(f'logs/{exp_name}/{load_exp}')['init_params']
    state = joblib.load(f'logs/{exp_name}/{load_exp}_state')['distribution']

    params['sigma_init'] = state
    params['ep_per_fit'] = 1
    params['quiet'] = False
    
    print(params)
    experiment(**params)
