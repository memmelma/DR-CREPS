from itertools import product
from experiment_launcher import Launcher
from mushroom_rl.environments import LQR
from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS
from constrained_REPS import REPS_CON
if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0
    lqr_dim = 9

    launcher = Launcher(exp_name=f'lqr_{lqr_dim}{"_env_" + str(env_seed) if env_seed >= 0 else ""}',
                        python_file='lqr_mi_el',
                        n_exp=25,
                        memory=5000,
                        days=0,
                        hours=5,
                        minutes=0,
                        seconds=0,
                        n_jobs=5,
                        use_timestamp=True)
    
    # launcher.add_default_params(k=lqr_dim, lqr_dim=lqr_dim, n_epochs=500, fit_per_epoch=1, ep_per_fit=50, env_seed=env_seed, sigma_init=2e-1, eps=0.5)
    launcher.add_default_params(k=lqr_dim, lqr_dim=lqr_dim, n_epochs=1000, fit_per_epoch=5, ep_per_fit=100, env_seed=env_seed, sigma_init=1e-3, eps=0.2)

    launcher.add_experiment(alg='REPS')
    launcher.add_experiment(alg='REPS_MI')

    # launcher.add_experiment(alg='REPS_CON')
    # launcher.add_experiment(alg='REPS_MI_CON')

    launcher.run(local, test)
