from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

	local = False
	test = False

	experiment_name = f'ship_optim'

	launcher = Launcher(experiment_name,
						'el_ship_optim',
						25,
						memory=3000,
						days=2,
						hours=0,
						minutes=0,
						seconds=0,
						conda_env='iprl',
						use_timestamp=False)
	
	launcher.add_default_params(sigma_init=7e-2, ep_per_fit=1, fit_per_epoch=1)

	n_samples = 5000
	n_epochs = n_samples
	launcher.add_experiment(alg='NM', n_epochs=n_epochs)
	launcher.add_experiment(alg='BFGS', n_epochs=n_epochs)

	print(experiment_name)
	print('experiments:', len(launcher._experiment_list))
	launcher.run(local, test)

	