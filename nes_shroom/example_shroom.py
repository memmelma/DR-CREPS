import torch
import numpy as np
from nes_shroom import CoreNES
from modules import ProMPNES, LinearRegressorNES

from mushroom_rl.environments.pybullet_envs.air_hockey import AirHockeyHit
from mushroom_rl.environments import ShipSteering, LQR, Gym
from mushroom_rl.features import Features
from mushroom_rl.features.tiles import Tiles

if __name__ == '__main__':
    n_samples = 2000
    n_rollout = 2
    population_size = 256
    n_step = n_samples // (population_size * n_rollout)
    sigma_init = 1e-0

    optim_lr = 0.02

    horizon = 50
    mdp = LQR.generate(dimensions=5, horizon=horizon, max_pos=1., max_action=1.)

    sigma_init = 1e-3
    features = None

    # horizon = 120
    # mdp = AirHockeyHit(horizon=horizon, debug_gui=0, table_boundary_terminate=True)
    # mdp.horizon = horizon
    input_size = mdp.info.observation_space.shape[0]
    output_size = mdp.info.action_space.shape[0]

    # high = [150, 150, np.pi]
    # low = [0, 0, -np.pi]
    # n_tiles = [5, 5, 6]
    # low = np.array(low, dtype=float)
    # high = np.array(high, dtype=float)
    # tilings = Tiles.generate(n_tilings=3, n_tiles=n_tiles, 
    #                         low=low, high=high)
    # features = Features(tilings=tilings)
    
    # sigma_init = 7e-2

    # mdp = ShipSteering()
    
    # input_size = features.size
    # output_size = mdp.info.action_space.shape[0]

    policy = ProMPNES(input_size, output_size, population_size=population_size, l_decay= 0.999, l2_decay=0.005, sigma=sigma_init, n_rollout=n_rollout, maxSteps=horizon, features=features)
    policy = LinearRegressorNES(input_size, output_size, population_size=population_size, l_decay=1., l2_decay=0., sigma=sigma_init, n_rollout=n_rollout, features=features)
    
    exp_name = 'nes_try'
    alg = 'nes'
    seed = 32
    nes = CoreNES(policy, mdp, optimizer=torch.optim.Adam, optimizer_lr=optim_lr,
                    n_step=n_step, seed=seed)

    nes.train(strat=alg)
    
    nes.log(exp_name=exp_name, alg=alg, seed=seed, results_dir='..')

    
    