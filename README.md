# Online Abstraction with MDP Homomorphisms for Deep Learning—source code

##  Setup

* Install Python >= 3.5.
* Install all packages listed in requirements.txt: `pip install -r requirements.txt`.
* I use Tensorflow 1.7 with CUDA 9.1 and cuDNN 7.1, any other setup might produce different results.

## Usage

### Train a deep Q-network

#### Discrete environments

```
python -m abstract.scripts.solve.puck_stack_n.dqn_branch 2 4 --max-time-steps 2500 --max-episodes 200 --learning-rate 0.0001 --batch-size 30
```

```
python -m abstract.scripts.solve.puck_stack_n.dqn_branch 3 4 --max-episodes 1000 --max-time-steps 20000 --exploration-fraction 0.25 --learning-rate 0.0001 --batch-size 30
```

#### Fully convolutional network for pseudo-continuous environments

### Transfer options between environmnets using MDP homomorphisms

#### Discrete environments

#### Pseudo-continuous environments

## Environments

* **envs/puck_stack**: stack N pucks in a discrete grid world
* **envs/puck_stack_subgoal**: make two stacks of N pucks in a continuous grid world
* **envs/continuous_puck_stack**: stack N pucks in a psedo-continuous environment
* **envs/continuous_two_stack**: make two stacks of N pucks in a pseudo-continuous environment
* **envs/continuous_component**: arrange N pucks so that they form a connected component
* **envs/continuous_stairs**: build stairs from 3 or 6 pucks