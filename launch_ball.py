from itertools import product
from experiment_launcher import Launcher

if __name__ == '__main__':

	local = False
	test = False

	experiment_name = f'ball_iprl_final_kl'
	
	launcher = Launcher(exp_name=experiment_name,
						python_file='ball_mi_el',
						n_exp=25,
						memory=5000,
						days=0,
						hours=24,
						minutes=0,
						seconds=0,
						n_jobs=-1,
						use_timestamp=True)
	
	launcher.add_default_params(n_epochs=500, fit_per_epoch=1, ep_per_fit=25, n_basis=20, horizon=750, sigma_init=1e-1, mi_type='regression', sample_type='percentage', eps=0.7)

	# ours vs ours
	# k
	# launcher.add_experiment(alg='ConstrainedREPS', sample_type=None, kappa=2)
	launcher.add_experiment(alg='ConstrainedREPSMI', k=5, sample_type=None, kappa=2)
	launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=2)
	launcher.add_experiment(alg='ConstrainedREPSMI', k=50, sample_type=None, kappa=2)
	launcher.add_experiment(alg='ConstrainedREPSMI', k=100, sample_type=None, kappa=2)
	launcher.add_experiment(alg='ConstrainedREPSMI', k=250, sample_type=None, kappa=2)
	launcher.add_experiment(alg='ConstrainedREPSMI', k=500, sample_type=None, kappa=2)

	# launcher.add_experiment(alg='RWR')
	# launcher.add_experiment(alg='REPS')

	# launcher.add_experiment(alg='REPS_MI', k=25) 
	# launcher.add_experiment(alg='REPS_MI', k=100)

	# launcher.add_experiment(alg='ConstrainedREPS', kappa=5)

	# launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, k=25, kappa=5)
	# launcher.add_experiment(alg='ConstrainedREPSMI', gamma=0.1, k=100, kappa=5)

	# launcher.add_experiment(alg='REPS')
	# launcher.add_experiment(alg='RWR', eps=0.5)
	# launcher.add_experiment(alg='RWR', eps=0.7)
	# launcher.add_experiment(alg='ConstrainedREPS')

	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=3, k=0.2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=3, k=0.4)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=3, k=0.6)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.8)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=3, k=25)

	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.4)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.6)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=0.8)
	# launcher.add_experiment(alg='ConstrainedREPSMI', bins=4, k=25)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=5, eps=0.7, sample_type='percentage', gamma=0.1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, eps=0.7, sample_type='percentage', gamma=0.1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=0.2, eps=0.7, sample_type='percentage', gamma=0.1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=0.6, eps=0.7, sample_type='percentage', gamma=0.1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=0.8, eps=0.7, sample_type='percentage', gamma=0.1)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=5, eps=0.7, sample_type='percentage', gamma=0.1, bins=3, mi_type='sample')
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, eps=0.7, sample_type='percentage', gamma=0.1, bins=3, mi_type='sample')
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=5, eps=0.7, sample_type='percentage', gamma=0.1, bins=4, mi_type='sample')
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, eps=0.7, sample_type='percentage', gamma=0.1, bins=4, mi_type='sample')

	# launcher.add_experiment(alg='REPS', sigma_init=1e-3)
	# launcher.add_experiment(alg='ConstrainedREPS', sigma_init=1e-1)
	# launcher.add_experiment(alg='ConstrainedREPSMI', sigma_init=1e-1)
	
	# launcher.add_experiment(alg='REPS', sigma_init=1e-1)
	# launcher.add_experiment(alg='MORE', sigma_init=1e-1)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='fixed', gamma=1e-5)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=0.1)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type='fixed', gamma=1e-5)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type='percentage', gamma=0.1)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=50)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=75)


	print(experiment_name)

	launcher.run(local, test)
