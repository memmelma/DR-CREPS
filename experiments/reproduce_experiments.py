from experiment_config import default_params

def reproduce_diag_lqr_experiments(launcher, local):

    if local:
        params = default_params()
    else:
        params = dict()

    params['env'] = 'LQR'
    params['lqr_dim'] = 10
    params['red_dim'] = 7
    params['sigma_init'] = 3e-1
    params['results_dir'] = 'results'
    params['fit_per_epoch'] = 1

    launcher.add_default_params(**params)

    launcher.add_experiment(
        alg='RWR', eps=0.2,
        distribution='diag',
        sample_strat=None,
        n_epochs=80, ep_per_fit=25
    )

    launcher.add_experiment(
        alg='PRO', eps=0.2,
        distribution='diag',
        sample_strat='PRO',
        n_epochs=80, ep_per_fit=25
    )

    launcher.add_experiment(
        alg='REPS', eps=0.4,
        distribution='diag',
        sample_strat=None,
        n_epochs=80, ep_per_fit=25
    )

    launcher.add_experiment(
        alg='REPS-PE', eps=0.4,
        distribution='diag',
        C='PCC', k=30,
        sample_strat='percentage', lambd=0.1,
        n_epochs=80, ep_per_fit=25
    )

    launcher.add_experiment(
        alg='REPS-PE', eps=0.4,
        distribution='diag',
        C='PCC', k=30,
        sample_strat='percentage', lambd=0.9,
        n_epochs=80, ep_per_fit=25
    )

    launcher.add_experiment(
        alg='CREPS', eps=2.5, kappa=6.0,
        distribution='diag',
        sample_strat=None,
        n_epochs=80, ep_per_fit=25
    )

    launcher.add_experiment(
        alg='CREPS-PE', eps=2.5, kappa=6.0,
        distribution='diag',
        C='PCC', k=30,
        sample_strat='percentage', lambd=0.1,
        n_epochs=80, ep_per_fit=25
    )

    launcher.add_experiment(
        alg='CREPS-PE', eps=2.5, kappa=6.0,
        distribution='diag',
        C='PCC', k=30,
        sample_strat='percentage', lambd=0.9,
        n_epochs=80, ep_per_fit=25
    )

    return launcher

def reproduce_lqr_experiments(launcher, local):

    if local:
        params = default_params()
    else:
        params = dict()

    params['env'] = 'LQR'
    params['lqr_dim'] = 10
    params['red_dim'] = 7
    params['sigma_init'] = 3e-1
    params['results_dir'] = 'results'
    params['fit_per_epoch'] = 1

    launcher.add_default_params(**params)

    launcher.add_experiment(
        alg='MORE', eps=4.7, kappa=17.,
        distribution='cholesky',
        n_epochs=20, ep_per_fit=250
    )

    launcher.add_experiment(
        alg='CREPS', eps=4.7, kappa=17.,
        distribution='cholesky',
        n_epochs=33, ep_per_fit=150
    )

    launcher.add_experiment(
        alg='NES', optim_lr=0.03,
        distribution='cholesky',
        n_rollout=2, population_size=100,
        n_epochs=25, ep_per_fit=200
    )

    launcher.add_experiment(
        alg='PPO', actor_lr=0.003, critic_lr=0.003,
        n_epochs=50, ep_per_fit=100
    )

    launcher.add_experiment(
        alg='TRPO', critic_lr=0.003, max_kl=1.0,
        n_epochs=50, ep_per_fit=100
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=4.7, kappa=17.,
        k=50, distribution='gdr',
        C='PCC', mi_estimator = 'regression',
        sample_strat='percentage', lambd=0.1,
        n_epochs=100, ep_per_fit=50
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=4.7, kappa=17.,
        k=50, distribution='gdr',
        C='PCC', mi_estimator = 'regression',
        sample_strat=None,
        n_epochs=100, ep_per_fit=50
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=4.7, kappa=17.,
        k=50, distribution='gdr',
        C='MI', mi_estimator = 'regression',
        sample_strat='percentage', lambd=0.1,
        n_epochs=100, ep_per_fit=50
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=4.7, kappa=17.,
        k=50, distribution='gdr',
        C='MI', mi_estimator = 'regression',
        sample_strat=None,
        n_epochs=100, ep_per_fit=50
    )

    return launcher

def reproduce_ship_steering(launcher, local):

    if local:
        params = default_params()
    else:
        params = dict()

    params['env'] = 'ShipSteering'
    params['n_tilings'] = 3
    params['sigma_init'] = 7e-2
    params['results_dir'] = 'results'
    params['fit_per_epoch'] = 1

    launcher.add_default_params(**params)

    launcher.add_experiment(
        alg='DR-REPS-PE', eps=0.5,
        k=100, distribution='gdr',
        C='PCC',
        sample_strat='percentage', lambd=0.1,
        n_epochs=14, ep_per_fit=250
    )

    launcher.add_experiment(
        alg='DR-REPS-PE', eps=0.5,
        k=100, distribution='gdr',
        C='MI', mi_estimator='regression',
        sample_strat='percentage', lambd=0.1,
        n_epochs=14, ep_per_fit=250
    )

    launcher.add_experiment(
        alg='MORE', eps=4.4, kappa=20.,
        distribution='cholesky',
        n_epochs=233, ep_per_fit=15
    )

    launcher.add_experiment(
        alg='CREPS', eps=2.4, kappa=20.,
        distribution='cholesky',
        n_epochs=233, ep_per_fit=15
    )
    
    launcher.add_experiment(
        alg='NES', optim_lr=0.03,
        distribution='cholesky',
        n_rollout=2, population_size=100,
        n_epochs=17, ep_per_fit=200
    )

    launcher.add_experiment(
        alg='PPO', actor_lr=0.0003, critic_lr=0.0003,
        n_epochs=35, ep_per_fit=100
    )

    launcher.add_experiment(
        alg='TRPO', critic_lr=0.03, max_kl=0.01,
        n_epochs=35, ep_per_fit=100
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=3.4, kappa=20.,
        k=200, distribution='gdr',
        C='PCC',
        sample_strat='percentage', lambd=0.1,
        n_epochs=233, ep_per_fit=15
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=3.4, kappa=20.,
        k=200, distribution='gdr',
        C='MI', mi_estimator='regression',
        sample_strat='percentage', lambd=0.1,
        n_epochs=233, ep_per_fit=15
    )

    return launcher

def reproduce_air_hockey(launcher, local):

    if local:
        params = default_params()
    else:
        params = dict()

    params['env'] = 'AirHockey'
    params['n_basis'] = 30
    params['horizon'] = 120
    params['sigma_init'] = 1e-0
    params['results_dir'] = 'results'
    params['fit_per_epoch'] = 1

    launcher.add_default_params(**params)

    launcher.add_experiment(
        alg='MORE', eps=2.4, kappa=12.,
        distribution='cholesky',
        n_epochs=40, ep_per_fit=250
    )

    launcher.add_experiment(
        alg='CREPS', eps=2.0, kappa=12.,
        distribution='cholesky',
        n_epochs=40, ep_per_fit=250
    )
    
    launcher.add_experiment(
        alg='NES', optim_lr=0.3,
        distribution='cholesky',
        n_rollout=2, population_size=100,
        n_epochs=50, ep_per_fit=200
    )

    launcher.add_experiment(
        alg='PPO', actor_lr=0.003, critic_lr=0.003,
        n_epochs=100, ep_per_fit=100
    )

    launcher.add_experiment(
        alg='TRPO', critic_lr=0.003, max_kl=0.1,
        n_epochs=100, ep_per_fit=100
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=2.0, kappa=12.,
        k=30, distribution='gdr',
        C='PCC',
        sample_strat='percentage', lambd=0.5,
        n_epochs=200, ep_per_fit=50
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=2.0, kappa=12.,
        k=30, distribution='gdr',
        C='MI', mi_estimator='regression',
        sample_strat='percentage', lambd=0.5,
        n_epochs=200, ep_per_fit=50
    )

    return launcher

def reproduce_ball_stopping(launcher, local):
    
    if local:
        params = default_params()
    else:
        params = dict()

    params['env'] = 'BallStopping'
    params['n_basis'] = 20
    params['horizon'] = 750
    params['sigma_init'] = 1e-0
    params['results_dir'] = 'results'
    params['fit_per_epoch'] = 1

    launcher.add_default_params(**params)

    launcher.add_experiment(
        alg='MORE', eps=4.5, kappa=20.,
        distribution='cholesky',
        n_epochs=28, ep_per_fit=250
    )

    launcher.add_experiment(
        alg='CREPS', eps=4.5, kappa=20.,
        distribution='cholesky',
        n_epochs=28, ep_per_fit=250
    )
    
    launcher.add_experiment(
        alg='NES', optim_lr=0.3,
        distribution='cholesky',
        n_rollout=2, population_size=100,
        n_epochs=35, ep_per_fit=200
    )

    launcher.add_experiment(
        alg='PPO', actor_lr=0.003, critic_lr=0.003,
        n_epochs=70, ep_per_fit=100
    )
    
    launcher.add_experiment(
        alg='TRPO', critic_lr=0.03, max_kl=1.0,
        n_epochs=70, ep_per_fit=100
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=4.5, kappa=20.,
        k=30, distribution='gdr',
        C='PCC',
        sample_strat='percentage', lambd=0.5,
        n_epochs=116, ep_per_fit=60
    )

    launcher.add_experiment(
        alg='DR-CREPS-PE', eps=4.5, kappa=20.,
        k=30, distribution='gdr',
        C='MI', mi_estimator='regression',
        sample_strat='percentage', lambd=0.5,
        n_epochs=116, ep_per_fit=60
    )

    return launcher