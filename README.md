# Online Abstraction with MDP Homomorphisms for Deep Learning—source code

##  Setup

* Install Python >= 3.5.
* Install all packages listed in requirements.txt: `pip install -r requirements.txt`.
* I use Tensorflow 1.7 with CUDA 9.1 and cuDNN 7.1, any other setup might produce different results.

## Usage

### Train a deep Q-network

### Transfer options between environmnets using MDP homomorphisms

## Environments

* **envs/puck_stack**: stack N pucks in a discrete grid world
* **envs/puck_stack_subgoal**: make two stacks of N pucks in a continuous grid world
* **envs/continuous_puck_stack**: stack N pucks in a psedo-continuous environment
* **envs/continuous_two_stack**: make two stacks of N pucks in a pseudo-continuous environment
* **envs/continuous_component**: arrange N pucks so that they form a connected component
* **envs/continuous_stairs**: build stairs from 3 or 6 pucks