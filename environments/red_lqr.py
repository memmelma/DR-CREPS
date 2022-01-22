import numpy as np

from mushroom_rl.environments import LQR
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain

def generate_red_LQR(lqr_dim, red_dim, horizon = 50, max_iterations=100000):
    
    mdp = LQR.generate(dimensions=lqr_dim, horizon=horizon, max_pos=1., max_action=1.)

    rng = np.random.default_rng(seed=0)
    choice = np.arange(0,lqr_dim,1)
    rng.shuffle(choice)

    ineff_params = choice[:red_dim]
    
    for p in ineff_params:
        mdp.B[p][p] = 1e-20
        mdp.Q[p][p] = 1e-20
    
    # compute optimal control return
    gain_lqr = compute_lqr_feedback_gain(mdp, max_iterations=max_iterations)
    state = mdp.reset()
    optimal_reward = 0
    for i in range(horizon):
        action = - gain_lqr @ state
        state, reward, _, __ = mdp.step(action)
        optimal_reward += reward
    
    return mdp, ineff_params, gain_lqr, optimal_reward