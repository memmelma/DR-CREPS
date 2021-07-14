from itertools import product
from math import gamma
from experiment_launcher import Launcher

if __name__ == '__main__':

	local = False
	test = False

	experiment_name = f'ball_eps_kappa'
	
	launcher = Launcher(exp_name=experiment_name,
						python_file='el_ball_mi',
						n_exp=25,
						memory=5000,
						days=2,
						hours=24,
						minutes=0,
						seconds=0,
						n_jobs=-1,
						use_timestamp=True)
	
	launcher.add_default_params(n_epochs=250, fit_per_epoch=1, ep_per_fit=25, n_basis=20, horizon=750, sigma_init=1e-0, mi_type='regression', sample_type='percentage')
	
	# eps = 0.7
	# kappa = 3.
	# launcher.add_experiment(alg='REPS', eps=eps)
	# launcher.add_experiment(alg='RWR', eps=eps)
	# launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa)
	# launcher.add_experiment(alg='ConstrainedREPSMI', eps=eps, k=25, gamma=1e-1, kappa=kappa)

	# eps and kappa:
	import numpy as np
	for eps in np.arange(0.2, 2.3, 0.5):
		eps = round(eps,1)
		# REPS
		launcher.add_experiment(alg='REPS', eps=eps)
		# RWR
		launcher.add_experiment(alg='RWR', eps=eps)
		
		for kappa in np.arange(2., 10., 2.):
			kappa = round(kappa,1)
			# Constrained REPS
			launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa)

	# k -> lqr w/ 10dim has 100 parameters
	# for k in range(5, 100, 5):
	# 	 eps = 4.0
	# 	 kappa = 6.0
	# 	 launcher.add_experiment(alg='ConstrainedREPSMI', eps=eps, kappa=kappa, sample_type=None, k=k)
		
	# 	 eps = 0.6
	# 	 launcher.add_experiment(alg='REPS_MI', eps=eps, sample_type=None, k=k)
	
	# n_epochs=500,
	
	# ours vs ours
	# k
	# launcher.add_experiment(alg='ConstrainedREPS', sample_type=None, kappa=2
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=5, sample_type=None, kappa=2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=25, sample_type=None, kappa=2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=50, sample_type=None, kappa=2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=100, sample_type=None, kappa=2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=250, sample_type=None, kappa=2)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=500, sample_type=None, kappa=2)

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
