from itertools import product
from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

	local = False
	test = False

	# experiment_name = f'ball_sample'
	experiment_name = f'ball_reinf'
	
	launcher = Launcher(experiment_name,
						'el_ball_rebuttal',
						25,
						memory=3000,
						days=3,
						hours=0,
						minutes=0,
						seconds=0,
						conda_env='iprl',
						use_timestamp=False)
	
	# launcher.add_default_params(fit_per_epoch=1, \
	# 							n_basis=20, horizon=750, \
	# 							sigma_init=1e-0, bins=4, mi_type='regression', mi_avg=0)
	
	# # all best
	# launcher.add_experiment(alg='ConstrainedREPS', eps=2.7, kappa=6.)
	# launcher.add_experiment(alg='REPS', eps=0.7)
	# launcher.add_experiment(alg='REPS_MI', eps=0.7, k=110, sample_type='percentage', gamma=0.9)
	# launcher.add_experiment(alg='ConstrainedREPSMI', eps=2.7, kappa=6., k=65, sample_type='percentage', gamma=0.5)
	# launcher.add_experiment(alg='RWR', eps=1.7)
	

	# eps and kappa:
	# for eps in np.arange(0.2, 3.3, 0.5):
	# 	eps = round(eps,1)
	# 	# REPS
	# 	launcher.add_experiment(alg='REPS', eps=eps)
	# 	# RWR
	# 	launcher.add_experiment(alg='RWR', eps=eps)
		
	# 	for kappa in np.arange(2., 18., 2.):
	# 		kappa = round(kappa,1)
	# 		# Constrained REPS
	# 		launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa)

	# # hyperparameters according to max reward
	# for k in range(5, 140, 15):
	# 	eps = 2.7
	# 	kappa = 6.0
	# 	launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type=None, eps=eps, kappa=kappa)
	# 	launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='importance', eps=eps, kappa=kappa)
	# 	for gama in np.arange(0.1, 1.0, 0.2):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='ConstrainedREPSMI', k=k, sample_type='percentage', gamma=gama, eps=eps, kappa=kappa)

	# 	eps = 0.7
	# 	# launcher.add_experiment(alg='REPS_MI', k=k, sample_type=None, eps=eps) # -> REPS
	# 	launcher.add_experiment(alg='REPS_MI', k=k, sample_type='importance', eps=eps)

	# 	for gama in np.arange(0.1, 1.0, 0.2):
	# 		gama = round(gama,1)
	# 		launcher.add_experiment(alg='REPS_MI', k=k, sample_type='percentage', gamma=gama, eps=eps)

	# # MI vs Pearson
	# launcher.add_experiment(alg='REPS_MI', k=110, method='Pearson', sample_type='percentage', gamma=0.9, eps=0.7)
	# launcher.add_experiment(alg='REPS_MI', k=110, method='Pearson', sample_type='PRO', eps=0.7)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=65, method='Pearson', sample_type='percentage', gamma=0.5, eps=2.7, kappa=6.)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=65, method='Pearson', sample_type='PRO', eps=2.7, kappa=6.)
	
	# launcher.add_experiment(alg='REPS_MI', k=110, method='MI', sample_type='percentage', gamma=0.9, eps=0.7)
	# launcher.add_experiment(alg='REPS_MI', k=110, method='MI', sample_type='PRO', eps=0.7)

	# launcher.add_experiment(alg='ConstrainedREPSMI', k=65, method='MI', sample_type='percentage', gamma=0.5, eps=2.7, kappa=6.)
	# launcher.add_experiment(alg='ConstrainedREPSMI', k=65, method='MI', sample_type='PRO', eps=2.7, kappa=6.)

	# n_samples = 7000
	# eps = 4.5
	# kappa = 20.

	# ep_per_fit = 250
	# n_epochs = n_samples // ep_per_fit
	# distribution = 'cholesky'
	# launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa, distribution=distribution, sample_type=None, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
	# launcher.add_experiment(alg='MORE', eps=eps, kappa=kappa, distribution=distribution, sample_type=None, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	

	
	# k = 30
	# ep_per_fit = 60
	# n_epochs = n_samples // ep_per_fit
	# distribution = 'mi'
	# for method in ['Pearson', 'MI', 'Random']:
	# 	for gama in [0.1, 0.5]:
		
	# 		launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, distribution=distribution, sample_type='percentage', method=method, gamma=gama, k=k, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	# 	launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, distribution=distribution, sample_type=None, method=method, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	launcher.add_default_params(fit_per_epoch=1, \
								n_basis=20, horizon=750, \
								sigma_init=1e-0)

	n_samples = 7000
	
	# # # for ep_per_fit in [50, 100, 150, 200, 250]:
	for ep_per_fit in [100]:
		n_epochs = n_samples // ep_per_fit
	# 	for eps in [3e-2, 3e-3, 3e-4]:		
	# 		launcher.add_experiment(alg='PPO', eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
	# 		for kappa in [1e-0, 1e-1, 1e-2]:
	# 			launcher.add_experiment(alg='TRPO', eps=eps, kappa=kappa, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
		for eps in [1e-1, 1e-2, 1e-3]:
			launcher.add_experiment(alg='REINFORCE', eps=eps, n_epochs=n_epochs, ep_per_fit=ep_per_fit)
	
	# ep_per_fit = 60
	# n_epochs = n_samples // ep_per_fit
	# launcher.add_experiment(alg='ConstrainedREPSMIFull', distribution='mi', eps=4.5, kappa=20., sample_type='percentage', method='Pearson', gamma=0.5, k=30, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

	
	# for ep_per_fit in [50, 100, 150, 200, 250]:
	# # for ep_per_fit in [20]:
	# 	n_epochs = n_samples // ep_per_fit
	# 	n_elites = ep_per_fit // 2
	# 	launcher.add_experiment(alg='CEM', eps=n_elites, n_epochs=n_epochs, ep_per_fit=ep_per_fit)

	# print(experiment_name)
	# print('experiments:', len(launcher._experiment_list))

	launcher.run(local, test)
