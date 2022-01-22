import argparse

policy_search = ['CEM', 'REPS', 'REPS-PE', 'DR-REPS-PE', 'RWR', 'PRO', 'RWR-PE',
                    'MORE', 'CREPS', 'CREPS-PE', 'DR-CREPS-PE']
gradient_methods = ['TRPO', 'PPO', 'REINFORCE']
evolution_strategies = ['ES', 'NES']
optimizers = ['NM', 'BFGS']

def experiment(**kwargs):
    if kwargs['env'] == 'LQR':
        if kwargs['alg'] in gradient_methods:
            from experiments.lqr.el_grad import experiment
        elif kwargs['alg'] in evolution_strategies:
            from experiments.lqr.el_es import experiment
        elif kwargs['alg'] in policy_search:
            from experiments.lqr.el_bbo import experiment
        elif kwargs['alg'] in optimizers:
            from experiments.lqr.el_optim import experiment
        experiment(**kwargs)

    elif kwargs['env'] == 'ShipSteering':
        experiment_ship_steering(**kwargs)
    elif kwargs['env'] == 'AirHockey':
        experiment_air_hockey(**kwargs)
    elif kwargs['env'] == 'BallStopping':
        experiment_ball_stopping(**kwargs)
    else:
        print('Environment does not exist! Exiting ...')


def default_params():
    defaults = dict(
        
        # environment
        env = 'LQR',
        seed = 0,
        env_seed = 42,

        # LQR
        lqr_dim = 10,
        red_dim = 7,

        # ShipSteering
        n_tilings = 3,

        # AirHockey and BallStopping: ProMP
        n_basis = 30,
        horizon = 750,

        # policy search
        alg = 'BFGS',
        eps = 1.,
        kappa = 3.5,
        # GDR
        k = 30,

        # gradient methods
        nn_policy = 0,
        # PPO
        actor_lr = 3e-3,
        # PPO/TRPO
        critic_lr = 3e-3,
        # TRPO
        max_kl = 1e-1,
        # REINFORCE
        optim_eps = 1e-2,

        # evolution strategies
        n_rollout = 10,
        population_size = 256,
        optim_lr = 1e-2,

        # distribution
        distribution = 'diag',
        sigma_init = 3e-1,

        # correlation measure
        C = 'PCC',
        mi_estimator = 'regression',

        # sample strategy
        sample_strat = None,
        # PE
        lambd = 0.1,

        # training
        n_epochs = 10,
        fit_per_epoch = 1,
        ep_per_fit = 50,

        # misc
        results_dir = 'results',
        save_render_path = None,
        verbose = 0
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()
    
    # environment
    parser.add_argument('--env', type=str, help="enviroment string; one of ['LQR', 'ShipSteering', 'AirHockey', 'BallStopping']")
    parser.add_argument('--seed', type=int, help="seed")
    parser.add_argument('--env-seed', type=int, help="environment seed")

    # LQR
    parser.add_argument('--lqr-dim', type=int, help="dimensionality of LQR")
    parser.add_argument('--red_dim', type=int, help="reduced dimensionalities of LQR")

    # ShipSteering
    parser.add_argument('--n-tilings', type=int, help="number of tilings to use as features in 'ShipSteering'")

    # AirHockey and BallStopping: ProMP
    parser.add_argument('--n-basis', type=int, help="number of Gaussian basis function for ProMP in 'AirHocket' and 'BallStopping'")
    parser.add_argument('--horizon', type=int, help="horizon of 'BallStopping")

    # policy search algorithm
    parser.add_argument('--alg', type=str, help=f"algorithm string; one of policy_search: {policy_search}, gradient_methods: {gradient_methods}, evolution_strategies: {evolution_strategies}, optimizers: {optimizers}")
    parser.add_argument('--eps', type=float, help="KL constraint for policy search algorithms")
    parser.add_argument('--kappa', type=float, help="Entropy constraint for policy search algorithms")

    # gradient methods
    parser.add_argument('--nn-policy', type=int, help="whether to use (1) or not to use (0) neural network policy in TRPO/PPO")
    parser.add_argument('--actor-lr', type=float, help="actor learning rate for PPO")
    parser.add_argument('--critic-lr', type=float, help="critic learning rate for PPO/TRPO")
    parser.add_argument('--max-kl', type=float, help="maximum KL for TRPO constraint")
    parser.add_argument('--optim-eps', type=float, help="eps for REINFORCE optimizer")

    # evolution strategies
    parser.add_argument('--n-rollout', type=int, help="number of rollouts")
    parser.add_argument('--population-size', type=int, help="population size")
    parser.add_argument('--optim-lr', type=float, help="optimizer learning rate")

    # distribution
    parser.add_argument('--distribution', type=str, help="distribution to use for policy search algorithms; one of ['diag', 'fixed', 'gdr', 'cholesky']")
    parser.add_argument('--sigma-init', type=float, help="standard deviation to initialize distribution")

    # correlation measure
    parser.add_argument('--C', type=str, help="correlation measure; one of ['MI', 'PCC', 'Random']")
    parser.add_argument('--mi-estimator', type=str, help="MI estimator to use; one of ['regression', 'score', 'hist']")
    
    # sample strategy
    parser.add_argument('--sample-strat', type=str, help="sample strategy; one of ['PRO', 'importance', 'fixed', 'percentage']")

    # Guided Dimensionality Reduction (GDR)
    parser.add_argument('--k', type=int, help="number of effective parameters to be selected")

    # Prioritized Exploration (PE)
    parser.add_argument('--lambd', type=float, help="discount factor \lambda for PE; --sample-strat must be set to 'percentage'")

    # training
    parser.add_argument('--n-epochs', type=int, help="number of epochs to train")
    parser.add_argument('--fit-per-epoch', type=int, help="fits per epoch")
    parser.add_argument('--ep-per-fit', type=int, help="episodes/rollouts per fit")

    # misc
    parser.add_argument('--results-dir', type=str, help="directory to save results")
    parser.add_argument('--save-render-path', type=str, help="video render path for Gym environments 'AirHockey' and 'BallStopping'")
    parser.add_argument('--verbose', type=int, help="verbose (1) or quite (0)")

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
