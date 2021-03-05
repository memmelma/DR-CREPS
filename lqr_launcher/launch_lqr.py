from itertools import product
from experiment_launcher import Launcher
from mushroom_rl.environments import LQR
from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS
from constrained_REPS import REPS_CON
if __name__ == '__main__':

    local = False
    test = True

    env_seed = 0
    lqr_dim = 7

    launcher = Launcher(exp_name=f'lqr_{lqr_dim}_env_{env_seed}',
                        python_file='lqr_mi_el',
                        n_exp=25,
                        memory=5000,
                        days=0,
                        hours=5,
                        minutes=0,
                        seconds=0,
                        n_jobs=4,
                        use_timestamp=True)
    
    launcher.add_default_params(eps=0.2, k=lqr_dim, lqr_dim=lqr_dim, n_epochs=500, fit_per_epoch=10, ep_per_fit=100, env_seed=env_seed)

    launcher.add_experiment(alg='REPS')
    launcher.add_experiment(alg='REPS_MI')
    launcher.add_experiment(alg='REPS_MI_CON')

    launcher.run(local, test)
