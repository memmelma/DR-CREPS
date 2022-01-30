import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import analytical_MI, compute_MI, get_mean_and_confidence

def plot_samples(m, n, samples=[5, 10, 15], bins=[3, 4], noise_factor=2, n_runs=25, log_dir='', random_seed=42, override=False):
	
	file_name = f'mi_m_{m}_n_{n}_bins_{bins}_{f"noise_{noise_factor}" if noise_factor > 0. else ""}'
	mi_runs = []

	if os.path.isfile(os.path.join(log_dir, 'data', file_name+'.npy')) and not override: 
		mi_runs, legend, colors, linestyle = np.load(os.path.join(log_dir, 'data', file_name+'.npy'), allow_pickle=True)
		print('Open existing file')
	else:
		for s in samples:
			mi_trial = []
			for i in range(n_runs):

				x, y , I, H_x = analytical_MI(noise_factor, m=m, n=n, samples=s, random_seed=random_seed+i)
				mi, legend, colors, linestyle = compute_MI(x, y, I, H_x, bins, random_seed=random_seed+i)
				
				mi_trial += [mi]
			mi_runs += [mi_trial]

		np.save(open(os.path.join(log_dir, 'data', file_name+'.npy'), 'wb'), (mi_runs, legend, colors, linestyle))

	fig, ax = plt.subplots()

	mean, ci = get_mean_and_confidence(np.transpose(mi_runs, (1,0,2)))

	for i in range(0, mean.shape[1]-1):
		print(colors[1:][i//len(bins)], i, i//len(bins))
		ax.plot(samples, mean[:,i],
				color=colors[1:][i//len(bins)], 
				linestyle=linestyle[i%len(bins)])
		ax.fill_between(samples, mean[:,i]+ci[0][:,i], mean[:,i]+ci[1][:,i], 
						color=colors[1:][i//len(bins)], alpha=.2)

	ax.plot(samples,  mean[:,0], color=colors[0], linestyle='-')

	legend_elements = []
	for color, leg in zip(colors, legend):
		legend_elements += [Line2D([0], [0], color=color, lw=1, label=leg)]
	
	if len(colors) % 2 != 0:
		legend_elements += [Line2D([0], [0], color='white', lw=1, label='')]

	for i, bin in enumerate(bins):
		legend_elements += [Line2D([0], [0], color='black', lw=1, label=f'bins/$k$={bin}', linestyle=linestyle[i])]


	ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2),
		  fancybox=True, ncol=4)
	plt.subplots_adjust(bottom=0.25)

	x_0, x_1 = samples[0], samples[-1]
	plt.xlim(x_0, x_1)
	# y_0, y_1 = -5, 5
	# plt.ylim(y_0, y_1)

	ax.set_xlabel('samples', fontsize=20)
	ax.set_ylabel('mutual information', fontsize=20)
	plt.title(f'$\Sigma_{{xx}} \in \mathbb{{R}}^{{{m}x{m}}}, \Sigma_{{yy}} \in \mathbb{{R}}^{{{n}x{n}}}, A \in \mathbb{{R}}^{{{n}x{m}}}$, {"w/" if bool(noise_factor) else "w/o"} noise', fontsize=20)

	plt.tight_layout()
	plt.grid()

	plt.savefig(os.path.join(log_dir, 'imgs', file_name+'.pdf'))
	plt.savefig(os.path.join(log_dir, 'imgs', file_name+'.png'))

if __name__ == '__main__':

	log_dir = 'logs_mi'
	img_dir = os.path.join(log_dir, 'imgs')
	data_dir = os.path.join(log_dir, 'data')
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(img_dir, exist_ok=True)
	os.makedirs(data_dir, exist_ok=True)

	plot_samples(m=10, n=1, samples=np.arange(10, 500, 25), bins=[3,4], noise_factor=0., n_runs=1, log_dir=log_dir, random_seed=42, override=True)

	# plot_samples(m=10, n=1, samples=np.arange(10, 500, 25), bins=[4], noise_factor=0., n_runs=25, log_dir=log_dir, random_seed=42, override=True)

	# plot_samples(m=100, n=1, samples=np.arange(10, 500, 25), bins=[3,4,5], noise_factor=0., n_runs=25, log_dir=log_dir, random_seed=42, override=True)
	
	# plot_samples(m=500, n=1, samples=np.arange(10, 500, 25), bins=[3,4,5], noise_factor=0., n_runs=25, log_dir=log_dir, random_seed=42, override=True)