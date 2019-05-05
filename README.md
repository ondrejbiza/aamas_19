# Online Abstraction with MDP Homomorphisms for Deep Learning—source code

This repository contains the source code to our [AAMAS'19 paper](https://arxiv.org/abs/1811.12929). 
The aim of the paper is to find abstractions
in the form of MDP homomorphisms based on experience collected by a Deep Reinforcement Learning agent.
We use a fully-convolutional deep Q-network to collect the experience.

##  Setup

* Install Python >= 3.5.
* Install all packages listed in requirements.txt: `pip install -r requirements.txt`.
* I use tensorflow-gpu 1.7 with CUDA 9.1 and cuDNN 7.1; any other setup might produce different results.

## Usage

### Train a deep Q-network

#### Discrete environments

Train a deep Q-network to stack 2 pucks in a grid world environment:
```
python -m abstract.scripts.solve.puck_stack_n.dqn_branch 2 4 
    --max-time-steps 2500 --max-episodes 200 --learning-rate 0.0001 --batch-size 30
```
Stacking three pucks:
```
python -m abstract.scripts.solve.puck_stack_n.dqn_branch 3 4 
    --max-episodes 1000 --max-time-steps 20000 --exploration-fraction 0.25 
    --learning-rate 0.0001 --batch-size 30
```

#### Fully convolutional network for pseudo-continuous environments

Train a deep Q-network on the continuous component task:
```
# 2, 3 or 4 pucks should work
num_pucks=2

python -m abstract.scripts.solve.continuous_component.dqn_fc 4 112 ${num_pucks} \
        --max-time-steps 400000 --max-episodes 15000 \
        --learning-rate 0.0001 --exploration-fraction 0.025 \
        --num-filters 32 64 64 32 --filter-sizes 8 8 3 1 --strides 4 2 1 1 \
        --upsample upsample_after
```
Building stairs:
```
# 3 or 6 pucks; the latter would require a lot more time steps (perhaps in the millions)
num_pucks=3

python -m abstract.scripts.solve.continuous_stairs.dqn_fc 4 112 ${num_pucks} \
        --max-time-steps 400000 --max-episodes 15000 \
        --learning-rate 0.0001 --exploration-fraction 0.025 \
        --num-filters 32 64 64 32 --filter-sizes 8 8 3 1 --strides 4 2 1 1 \
        --upsample upsample_after
```
Stacking pucks:
```
# 2 or 3 pucks; stacking 4 pucks would require a lot of time steps
num_pucks=2

python -m abstract.scripts.solve.continuous_puck_stack_n.dqn_fc 4 112 ${num_pucks} \
        --max-time-steps 400000 --max-episodes 15000 \
        --learning-rate 0.0001 --exploration-fraction 0.025 \
        --num-filters 32 64 64 32 --filter-sizes 8 8 3 1 --strides 4 2 1 1 \
        --upsample upsample_after
```

### Collect data for the abstraction algorithm

#### Discrete environment

The transfer script for the discrete environment collects the initial experience during each run.

#### Pseudo-continuous environments
You need to collect the data for abstraction using the following shell scripts:

```
./abstract/shell_scripts/abstraction/continuous_component/collect_data_dqn.sh
./abstract/shell_scripts/abstraction/continuous_puck_stack_n/collect_data_dqn.sh
./abstract/shell_scripts/abstraction/continuous_stairs/collect_data_dqn.sh
```

### Transfer options between environments using MDP homomorphisms

#### Discrete environments

Transfer from 2 to 3 pucks stacking in a grid world environment:
```
# transfer options
python -m abstract.scripts.abstract.puck_stack_n.dqn_exp_goal_transfer 4 1 --num-pucks-list 2 3 \
        --num-start-episodes 1000 --num-episodes 0 --max-buffer-size 10000 \
        --min-radius 7 --max-radius 12 --reuse --max-blocks 10 \
        --reward-threshold 0.98 --early-stop 1000 --softmax-selection --no-sharing \
        --dqn-final-epsilon 0.1 --dqn-num-exp-steps 5000 --state-action-threshold 400

# transfer weights
python -m abstract.scripts.abstract.puck_stack_n.dqn_exp_goal_transfer 4 1 --num-pucks-list 2 3 \
        --num-start-episodes 1000 --num-episodes 0 --max-buffer-size 10000 \
        --min-radius 7 --max-radius 12 --reuse --no-sharing \
        --dqn-final-epsilon 0.1 --dqn-num-exp-steps 5000 --no-option \
        --share-dqn --share-dqn-reset-buffer
```

Transfer from 3 pucks stacking to 2 and 2 puck stacking in a grid world environment:
```
# transfer options
python -m scripts.abstract.puck_stack_subgoal.dqn_exp_option_transfer 4 1 \
        --num-start-episodes 1500 --num-episodes 0 --max-buffer-size 10000 \
        --min-radius 7 --max-radius 12 --reuse --max-blocks 10 \
        --reward-threshold 0.98 --early-stop 1000 --softmax-selection --no-sharing \
        --dqn-final-epsilon 0.1 --dqn-num-exp-steps 10000 \
        --state-action-threshold 600 --option-learning-rate 0.1

# transfer weights
python -m scripts.abstract.puck_stack_subgoal.dqn_exp_option_transfer 4 1 \
        --num-start-episodes 1500 --num-episodes 0 --max-buffer-size 10000 \
        --min-radius 7 --max-radius 12 --reuse --max-blocks 10 \
        --reward-threshold 0.98 --early-stop 1000 --softmax-selection --no-sharing \
        --dqn-final-epsilon 0.1 --dqn-num-exp-steps 10000 --share-dqn \
        --no-option --share-dqn-reset-buffer
```

#### Pseudo-continuous environments

We ran many transfer experiments in the pseudo-continuous environments. The following is one example:
```
# transfer from 2 puck stacking to 3 component

# transfer options
python -m scripts.abstract.continuous_component.transfer_drn "dataset/dqn/continuous_puck_stack_2_112x112.pickle" \
        3 1000 10 --deduplicate --max-time-steps 400000 \
        --max-episodes 15000 --learning-rate 0.0001 --exploration-fraction 0.025 \
        --num-filters 32 64 64 32 --filter-sizes 8 8 3 1 --strides 4 2 1 1 \
        --upsample upsample_after --proportional-selection

# transfer weights
python -m scripts.solve.continuous_component.dqn_fc 3 112 3 \
        --max-time-steps 400000 --max-episodes 15000 \
        --learning-rate 0.0001 --exploration-fraction 0.025 \
        --num-filters 32 64 64 32 --filter-sizes 8 8 3 1 --strides 4 2 1 1 \
        --upsample upsample_after --load-weights "dataset/dqn/continuous_puck_stack_2_112x112"
```

## Environments

* **envs/puck_stack**: stack N pucks in a discrete grid world
* **envs/puck_stack_subgoal**: make two stacks of N pucks in a continuous grid world
* **envs/continuous_puck_stack**: stack N pucks in a psedo-continuous environment
* **envs/continuous_two_stack**: make two stacks of N pucks in a pseudo-continuous environment
* **envs/continuous_component**: arrange N pucks so that they form a connected component
* **envs/continuous_stairs**: build stairs from 3 or 6 pucks

## Authors

[Ondrej Biza](https://sites.google.com/view/obiza), supervised by [Robert Platt](http://www.ccs.neu.edu/home/rplatt/).