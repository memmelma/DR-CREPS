from platform import dist
from experiment_launcher import Launcher
import numpy as np

if __name__ == '__main__':

	local = False
	test = False

	experiment_name = f'hockey_ablation_random_fix_06'
	
	launcher = Launcher(experiment_name,
						'el_air_hockey_mi',
						25,
						memory=1000,
						days=1,
						hours=0,
						minutes=0,
						seconds=0,
						use_timestamp=False)
	
	launcher.add_default_params(n_basis=30, horizon=120,
								sigma_init=1e-0,
								bins=4, mi_type='regression', mi_avg=0, nn=0)
	

	n_samples = 10000

	## FULL COVARIANCE

	# distribution = 'cholesky'
	# ep_per_fit = 250
	# n_epochs = n_samples // ep_per_fit

	# # REPS
	# for eps in np.arange(0.1, 2.0, 0.2):
	# 	launcher.add_experiment(alg='REPS', eps=eps, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	# # RWR		
	# for eps in np.arange(0.1, 2.0, 0.2):
	# 	launcher.add_experiment(alg='RWR', eps=eps, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	# # Constrained REPS
	# for eps in np.arange(0.5, 3.3, 0.5):
	# 	for kappa in np.arange(6., 20., 2.):
	# 		kappa = round(kappa,1)
	# 		launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	# # MORE
	# # for eps in np.arange(1.2, 3.3, 1.):
	# for ep_per_fit in [200]:
	# 	for eps in np.arange(5.3, 8.3, 1.):
	# 		# for kappa in np.arange(5., 21., 5.):
	# 		for kappa in [15., 20.]:
	# 			kappa = round(kappa,1)
	# 			launcher.add_experiment(alg='MORE', eps=eps, kappa=kappa, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	# # #	# REPS MI
	# distribution = 'mi'
	# # # for ep_per_fit in [150, 250]:
	# for ep_per_fit in [200]:
	# 	n_epochs = n_samples // ep_per_fit
	# 	for eps in [0.3, 0.4]:
	# 		# for k in [30, 50, 70]:
	# 		for k in [50]:
	# 			# for method in ['MI', 'Random', 'Pearson']:
	# 			for method in ['MI', 'Pearson']:
	# 				# for gama in [0.1, 0.5, 0.9]:
	# 				for gama in [0.5, 0.9]:
	# 					launcher.add_experiment(alg='REPS_MI_full', eps=eps, k=k, sample_type='percentage', gamma=gama, method=method, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
	# 				launcher.add_experiment(alg='REPS_MI_full', eps=eps, k=k, sample_type=None, gamma=gama, method=method, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
		
	# #	# Constrained REPS MI
	# distribution = 'mi'
	# # for ep_per_fit in [100, 150, 250]:
	# for ep_per_fit in [50]:
	# 	n_epochs = n_samples // ep_per_fit
	# 	# for eps in [1.2, 1.4, 1.6]:
	# 	for eps in [2.0]: # 2.2
	# 		kappa = 12. # 15.
	# 		for k in [30]:
	# 			# for method in ['MI', 'Random', 'Pearson']:
	# 			for method in ['MI', 'Pearson']:
	# 				# for gama in [0.1, 0.5, 0.9]:
	# 				for gama in [0.5]:
	# 					launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama, method=method, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
	# 				# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, k=k, sample_type=None, gamma=gama, method=method, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
	
	# distribution = 'cholesky'
	# ep_per_fit = 250
	# n_epochs = n_samples // ep_per_fit
	# eps = 2.
	# kappa = 12.
	# launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
	# eps = 2.4
	# kappa = 12.
	# launcher.add_experiment(alg='MORE', eps=eps, kappa=kappa, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
	
	## DIAG COVARIANCE

	# distribution = 'diag'
	# for ep_per_fit in [25, 50, 100]:
	# 	n_epochs = n_samples // ep_per_fit
	
	# # REPS
	# for eps in np.arange(0.1, 2.0, 0.2):
	# 	launcher.add_experiment(alg='REPS', eps=eps, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
	
	# # RWR
	# for eps in np.arange(0.1, 2.0, 0.2):
	# 	launcher.add_experiment(alg='RWR', eps=eps, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
	
	# # Constrained REPS
	# for eps in np.arange(1.2, 3.3, 0.5):
	# 	for kappa in np.arange(3., 15., 2.):
	# 		kappa = round(kappa,1)
	# 		launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	# # REPS MI
	# distribution = 'diag'
	# for ep_per_fit in [25, 50, 100]:
	# 	n_epochs = n_samples // ep_per_fit
	# 	eps = 0.5
	# 	for k in [30, 50, 70]:
	# 		for method in ['MI', 'Random', 'Pearson']:
	# 			for gama in [0.1, 0.5, 0.9]:
	# 				launcher.add_experiment(alg='REPS_MI', eps=eps, k=k, sample_type='percentage', gamma=gama, method=method, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
		
	# # Constrained REPS MI
	# distribution = 'diag'
	# for ep_per_fit in [50, 100, 200]:
	# 	n_epochs = n_samples // ep_per_fit
	# 	eps = 2.2
	# 	kappa = 15.
	# 	for k in [30, 50, 70]:
	# 		for method in ['MI', 'Random', 'Pearson']:
	# 			for gama in [0.1, 0.5, 0.9]:
	# 				launcher.add_experiment(alg='ConstrainedREPSMI', eps=eps, kappa=kappa, k=k, sample_type='percentage', gamma=gama, method=method, distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	# # PRO
		# launcher.add_experiment(alg='ConstrainedREPSMI', eps=eps, kappa=kappa, sample_type='PRO', method='Pearson', distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
		# launcher.add_experiment(alg='REPS_MI', eps=eps, sample_type='PRO', method='Pearson', distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
		# launcher.add_experiment(alg='RWR_MI', eps=eps, sample_type='PRO', method='Pearson', distribution=distribution, ep_per_fit=ep_per_fit, n_epochs=n_epochs)


	# for ep_per_fit in [50, 150, 250]:
	# 	n_epochs = 10000 // ep_per_fit
	# 	k = 30
	# 	for eps in [3.0, 4.0]:
	# 		for kappa in [7.0, 15.]:
	# 			launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, method='Pearson', sample_type='percentage', gamma=0.1, k=k, distribution='mi', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
	# 			launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, method='MI', sample_type='percentage', gamma=0.1, k=k, distribution='mi', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
	# 			launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa, sample_type=None, distribution='cholesky', n_epochs=n_epochs, ep_per_fit=ep_per_fit)
	
	# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=5.0, kappa=7.0, method='Pearson', sample_type='percentage', gamma=0.1, k=30, distribution='mi', n_epochs=10000//50, ep_per_fit=25, nn=1)
	# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=4.0, kappa=7.0, method='Pearson', sample_type='percentage', gamma=0.1, k=30, distribution='mi', n_epochs=10000//50, ep_per_fit=25, nn=1)
	# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=4.0, kappa=7.0, method='Pearson', sample_type='percentage', gamma=0.5, k=30, distribution='mi', n_epochs=10000//50, ep_per_fit=25, nn=1)
	# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=4.0, kappa=7.0, method='Pearson', sample_type='percentage', gamma=0.1, k=30, distribution='mi', n_epochs=10000//50, ep_per_fit=50, nn=1)
	# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=4.0, kappa=7.0, method='Pearson', sample_type='percentage', gamma=0.5, k=30, distribution='mi', n_epochs=10000//50, ep_per_fit=50, nn=1)
	# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=4.0, kappa=7.0, method='Pearson', sample_type='percentage', gamma=0.1, k=30, distribution='mi', n_epochs=10000//50, ep_per_fit=150, nn=1)
	# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=4.0, kappa=7.0, method='Pearson', sample_type='percentage', gamma=0.5, k=30, distribution='mi', n_epochs=10000//50, ep_per_fit=150, nn=1)
		
	eps = 2.0
	kappa = 12.
	k = 30
	for ep_per_fit in [50,100]:
		n_epochs = int(200*50 / ep_per_fit)
		distribution = 'mi'
		launcher.add_experiment(alg='ConstrainedREPS', eps=eps, kappa=kappa, distribution='cholesky', sample_type=None, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
		for method in ['Pearson', 'MI', 'Random']:
			for gama in [0.1,0.9]:
				launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, distribution=distribution, sample_type='percentage', method=method, gamma=gama, k=k, ep_per_fit=ep_per_fit, n_epochs=n_epochs)
			# launcher.add_experiment(alg='ConstrainedREPSMIFull', eps=eps, kappa=kappa, distribution=distribution, sample_type=None, method=method, ep_per_fit=ep_per_fit, n_epochs=n_epochs)

	print(experiment_name)
	print('experiments:', len(launcher._experiment_list))

	launcher.run(local, test)
