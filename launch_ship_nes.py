from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

	local = False
	test = False

	experiment_name = f'ship_nes_fix'
	
	launcher = Launcher(experiment_name,
						'el_ship_nes',
						25,
						memory=3000,
						days=2,
						hours=0,
						minutes=0,
						seconds=0,
						conda_env='iprl',
						use_timestamp=False)

	launcher.add_default_params(n_tilings=3,
								sigma_init=7e-2, 
								fit_per_epoch=1)

	n_samples = 3500

	# for population_size in [50, 100, 200, 500]:
	for population_size in [100]:
		n_rollout = 2

		ep_per_fit = (population_size * n_rollout)
		n_epochs = n_samples // ep_per_fit

		for optim_lr in [3e-1, 3e-2, 3e-3]:
		# for optim_lr in [3e-4, 3., 6.]:
			launcher.add_experiment(alg='NES', optim_lr=optim_lr, n_rollout=n_rollout, population_size=population_size, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
			launcher.add_experiment(alg='ES', optim_lr=optim_lr, n_rollout=n_rollout, population_size=population_size, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

	print(experiment_name)
	print('experiments:', len(launcher._experiment_list))
	launcher.run(local, test)