from itertools import product
from experiment_launcher import Launcher
from mushroom_rl.algorithms.policy_search.black_box_optimization.reps import REPS
from constrained_REPS import REPS_CON
if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0
    lqr_dim = 10
    eff = 2
    k = eff*lqr_dim

    experiment_name = f'lqr_dim_{lqr_dim}_eff_{eff}{"_env_" + str(env_seed) if env_seed >= 0 else ""}'

    launcher = Launcher(exp_name=experiment_name,
                        python_file='lqr_mi_el',
                        n_exp=25,
                        memory=5000,
                        days=0,
                        hours=5,
                        minutes=0,
                        seconds=0,
                        n_jobs=5,
                        use_timestamp=True)
    
    launcher.add_default_params(k=k, lqr_dim=lqr_dim, n_epochs=1500, fit_per_epoch=1, ep_per_fit=100, env_seed=env_seed, n_ineff=lqr_dim-eff, eps=3e-3, sigma_init=0.15)

    launcher.add_experiment(alg='REPS')
    launcher.add_experiment(alg='REPS_MI_ORACLE')
    # launcher.add_experiment(alg='REPS_MI_FIXED_LOW')
    # launcher.add_experiment(alg='REPS_MI_FIXED_HIGH', kappa=1e-2)
    # launcher.add_experiment(alg='REPS_MI_10', kappa=0.1)

    print(experiment_name)

    launcher.run(local, test)
