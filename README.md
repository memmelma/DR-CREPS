# Dimensionality Reduction and Prioritized Exploration for Policy Search
by Memmel M., Liu P., Tateo D., Peters J.

[Abstract] Black-box policy optimization, a class of reinforcement learning algorithms, explores and updates policies at the parameter level. These types of algorithms are applied widely in robotics applications with movement primitives or non-differentiable policies and are particularly relevant where exploration at the action level could cause actuator damage or other safety issues. However, they do not scale well with the increasing dimensionality of the policy, triggering high demand for samples, expensive to obtain on real-world systems. In most setups, policy parameters do not contribute equally to the return. Thus, identifying the parameters contributing most allows us to narrow down the exploration and speed up the learning. Furthermore, updating only the effective parameters requires fewer samples and thereby solves the scalability issue. We present a novel method to prioritize the exploration of effective parameters and cope with full covariance matrix updates. Our algorithm learns faster than recent approaches and requires fewer samples to achieve state-of-the-art results. To select the effective parameters, we consider both the Pearson correlation coefficient and the Mutual Information. We showcase the capabilities of our approach on the Relative Entropy Policy Search algorithm in several simulated environments, including robotics simulations.

### requirements
```
git clone https://github.com/MushroomRL/mushroom-rl.git
cd mushroom-rl
git checkout dev
git checkout a0eaa2cf8001e433419234a9fc48b64170e3f61c
pip install -e .

git clone https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher.git
cd experiment_launcher
pip install -e .
```
