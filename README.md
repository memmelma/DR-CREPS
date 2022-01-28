# Dimensionality Reduction and Prioritized Exploration for Policy Search
by Memmel M., Liu P., Tateo D., Peters J.

[Abstract] _Black-box policy optimization, a class of reinforcement learning algorithms, explores and updates policies at the parameter level. These types of algorithms are applied widely in robotics applications with movement primitives or non-differentiable policies and are particularly relevant where exploration at the action level could cause actuator damage or other safety issues. However, they do not scale well with the increasing dimensionality of the policy, triggering high demand for samples, expensive to obtain on real-world systems. In most setups, policy parameters do not contribute equally to the return. Thus, identifying the parameters contributing most allows us to narrow down the exploration and speed up the learning. Furthermore, updating only the effective parameters requires fewer samples and thereby solves the scalability issue. We present a novel method to prioritize the exploration of effective parameters and cope with full covariance matrix updates. Our algorithm learns faster than recent approaches and requires fewer samples to achieve state-of-the-art results. To select the effective parameters, we consider both the Pearson correlation coefficient and the Mutual Information. We showcase the capabilities of our approach on the Relative Entropy Policy Search algorithm in several simulated environments, including robotics simulations._

### Changes made to Mushroom RL
```
mushroom.distributions.gaussian.GaussianCholeskyDistribution.con_wmle
eta_omg_opt_start = np.array([1., 1.])
scipy.minimize(..., method=None)
```

### Mushroom RL
Our implementation builds on [MushroomRL](https://github.com/MushroomRL/mushroom-rl). It also integrates nicely with it, i.e., all of our algorithms and environments either inherit directly from MushroomRL modules or implement the same intuitive interfaces.

### Getting Started
To run you own examples, start by setting up an environment with the following packages:

_MushroomRL-v.1.7.0_
```
git clone https://github.com/MushroomRL/mushroom-rl.git
cd mushroom-rl
git checkout dev
git checkout a0eaa2cf8001e433419234a9fc48b64170e3f61c
pip install -r requirements.txt
pip install -e .
```
_experiment_launcher_
```
git clone https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher.git
cd experiment_launcher
pip install -e .
```
You can then specify an experiment in the ```run.py``` and execute via ```python run.py```. Note that if you do not run on a cluster that supports SLURM or Joblib you might have to set ```local=True``` for a successful execution.

### Single Experiment
To run a single experiment directly execute ```experiment_config.py``` and specify the parameters via the command line or directly in the file by overriding default parameter dictionary. The script will then call the scripts in ```experiments/``` that correspond to the given algorithm and environment setup.

### Set Up Your Own Environment/Experiments
To get an idea of how to setup more detailed experiments please have a look at ```experiments/ENVIRONMENT/``` where you can find scripts for the following algorithm types:
- ```el_es.py```: natural evolution strategies (ES, NES)
- ```el_grad.py```: gradient based methods (REINFORCE, TRPO, PPO)
- ```el_optim.py```: classic optimizers (Nelder-Mead, L-BFGS-B)
- ```el_bbo.py```: policy search algorithms (RWR, REPS, CREPS, CEM, MORE, **DR-REPS**, **DR-CREPS**, PRO)
