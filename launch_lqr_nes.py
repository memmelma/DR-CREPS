from experiment_launcher import Launcher

if __name__ == '__main__':

    local = False
    test = False

    env_seed = 0
    lqr_dim = 10
    eff = 3

    experiment_name = f'lqr_nes_fix'

    launcher = Launcher(experiment_name,
                        'el_lqr_nes',
                        25,
                        memory=500,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=False,
                        conda_env='iprl')

    launcher.add_default_params(lqr_dim=lqr_dim, n_ineff=lqr_dim-eff, env_seed=env_seed,
                                fit_per_epoch=1,
                                sigma_init=3e-1)

    n_samples = 5000

    # for population_size in [50, 100, 200, 500]:
    for population_size in [100]:
        n_rollout = 2

        ep_per_fit = (population_size * n_rollout)
        n_epochs = n_samples // ep_per_fit

        for optim_lr in [3e-1, 3e-2, 3e-3]:
            launcher.add_experiment(alg='NES', optim_lr=optim_lr, n_rollout=n_rollout, population_size=population_size, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
            launcher.add_experiment(alg='ES', optim_lr=optim_lr, n_rollout=n_rollout, population_size=population_size, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
        # for optim_lr in [3e-1, 3e-2, 3e-3]:
        #     launcher.add_experiment(alg='NES', optim_lr=optim_lr, n_rollout=n_rollout, population_size=population_size, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
        #     launcher.add_experiment(alg='ES', optim_lr=optim_lr, n_rollout=n_rollout, population_size=population_size, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

    print(experiment_name)
    print('experiments:', len(launcher._experiment_list))
    launcher.run(local, test)

