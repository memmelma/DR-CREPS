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
        if kwargs['alg'] in gradient_methods:
            from experiments.ship_steering.el_grad import experiment
        elif kwargs['alg'] in evolution_strategies:
            from experiments.ship_steering.el_es import experiment
        elif kwargs['alg'] in policy_search:
            from experiments.ship_steering.el_bbo import experiment
        elif kwargs['alg'] in optimizers:
            from experiments.ship_steering.el_optim import experiment
        experiment(**kwargs)

    elif kwargs['env'] == 'AirHockey':
        if kwargs['alg'] in gradient_methods:
            from experiments.air_hockey.el_grad import experiment
        elif kwargs['alg'] in evolution_strategies:
            from experiments.air_hockey.el_es import experiment
        elif kwargs['alg'] in policy_search:
            from experiments.air_hockey.el_bbo import experiment
        elif kwargs['alg'] in optimizers:
            from experiments.air_hockey.el_optim import experiment
        experiment(**kwargs)

    elif kwargs['env'] == 'BallStopping':
        if kwargs['alg'] in gradient_methods:
            from experiments.ball_stopping.el_grad import experiment
        elif kwargs['alg'] in evolution_strategies:
            from experiments.ball_stopping.el_es import experiment
        elif kwargs['alg'] in policy_search:
            from experiments.ball_stopping.el_bbo import experiment
        elif kwargs['alg'] in optimizers:
            from experiments.ball_stopping.el_optim import experiment
        experiment(**kwargs)

    else:
        print('Environment does not exist! Exiting ...')


def default_params():
    defaults = dict(
        
        # environment
        env = 'BallStopping',
        seed = 0,
        env_seed = 42,

        # LQR
        lqr_dim = 0,
        red_dim = 0,

        # ShipSteering
        n_tilings = 0,

        # AirHockey and BallStopping: ProMP
        n_basis = 0,
        horizon = 0,

        # policy search
        alg = None,
        eps = 0,
        kappa = 0,
        # GDR
        k = 0,

        # gradient methods
        nn_policy = 0,
        # PPO
        actor_lr = 0,
        # PPO/TRPO
        critic_lr = 0,
        # TRPO
        max_kl = 0,
        # REINFORCE
        optim_eps = 0,

        # evolution strategies
        n_rollout = 0,
        population_size = 0,
        optim_lr = 0,

        # distribution
        distribution = None,
        sigma_init = 0,

        # correlation measure
        C = None,
        mi_estimator = None,

        # sample strategy
        sample_strat = None,
        # PE
        lambd = 0,

        # training
        n_epochs = 0,
        fit_per_epoch = 0,
        ep_per_fit = 0,

        # misc
        results_dir = None,
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
