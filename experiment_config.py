import argparse

from experiments.lqr import experiment_lqr
from experiments.ship_steering import experiment_ship_steering
from experiments.air_hockey import experiment_air_hockey
from experiments.ball_stopping import experiment_ball_stopping


def experiment(**kwargs):
    if kwargs['env'] == 'LQR':
        experiment_lqr(**kwargs)
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
        env = 'AirHockey',
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

        # algorithm
        alg = 'REPS',
        eps = 1.,
        kappa = 3.5,
        k = 30,

        # distribution
        distribution = 'diag',
        sigma_init = 3e-1,

        # correlation measure
        C = 'PCC',
        mi_estimator = 'regression',

        # sample strategy
        sample_strat = 'PRO',
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
    parser.add_argument('--env', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--env-seed', type=int)

    # LQR
    parser.add_argument('--lqr-dim', type=int)
    parser.add_argument('--red_dim', type=int)

    # ShipSteering
    parser.add_argument('--n-tilings', type=int)

    # AirHockey and BallStopping: ProMP
    parser.add_argument('--n-basis', type=int)
    parser.add_argument('--horizon', type=int)

    # algorithm
    parser.add_argument('--alg', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--k', type=int)
    
    # distribution
    parser.add_argument('--distribution', type=str)
    parser.add_argument('--sigma-init', type=float)

    # correlation measure
    parser.add_argument('--C', type=str)
    parser.add_argument('--mi-estimator', type=str)
    
    # sample strategy
    parser.add_argument('--sample-strat', type=str)
    parser.add_argument('--lambd', type=float)

    # training
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--fit-per-epoch', type=int)
    parser.add_argument('--ep-per-fit', type=int)

    # misc
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--save-render-path', type=str)
    parser.add_argument('--verbose', type=int)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
