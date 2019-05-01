#!/usr/bin/env bash

# 2 component, 1 redundant puck, 112x112 world and actions
action=112
num_blocks=4
num_pucks=2

num_runs=100
max_time_steps=800000
max_episodes=15000
exploration_fraction=0.05

num_filters="32 64 64 32"
filter_sizes="8 8 3 1"
strides="4 2 1 1"
upsample="upsample_after"
learning_rate=0.0001

save_exp_path="dataset/dqn/continuous_component_2_${num_pucks}_${action}x${action}_1_redundant.pickle"
save_weights_path="dataset/dqn/continuous_component_2_${num_pucks}_${action}x${action}_1_redundant"
save_rewards_path="dataset/dqn/continuous_component_2_${num_pucks}_${action}x${action}_1_redundant.dat"
save_exp_num=20000

if [[ ! -f "$save_exp_path" ]]; then
    python -m scripts.solve.continuous_component.dqn_fc "$num_blocks" "$action" "$num_pucks" \
            --max-time-steps "$max_time_steps" --max-episodes "$max_episodes" \
            --learning-rate "$learning_rate" --exploration-fraction "$exploration_fraction" \
            --num-filters ${num_filters} --filter-sizes ${filter_sizes} --strides ${strides} \
            --upsample ${upsample} --save-exp-path "$save_exp_path" --save-exp-num "$save_exp_num" \
            --save-weights "$save_weights_path" --num-redundant-pucks 1 --rewards-file "$save_rewards_path"
fi

# 3 component, 112x112 world and actions
action=112
num_blocks=4
num_pucks=3

num_runs=100
max_time_steps=800000
max_episodes=15000
exploration_fraction=0.05

num_filters="32 64 64 32"
filter_sizes="8 8 3 1"
strides="4 2 1 1"
upsample="upsample_after"
learning_rate=0.0001

save_exp_path="dataset/dqn/continuous_component_3_${num_pucks}_${action}x${action}.pickle"
save_weights_path="dataset/dqn/continuous_component_3_${num_pucks}_${action}x${action}"
save_rewards_path="dataset/dqn/continuous_component_3_${num_pucks}_${action}x${action}.dat"
save_exp_num=20000

if [[ ! -f "$save_exp_path" ]]; then
    python -m scripts.solve.continuous_component.dqn_fc "$num_blocks" "$action" "$num_pucks" \
            --max-time-steps "$max_time_steps" --max-episodes "$max_episodes" \
            --learning-rate "$learning_rate" --exploration-fraction "$exploration_fraction" \
            --num-filters ${num_filters} --filter-sizes ${filter_sizes} --strides ${strides} \
            --upsample ${upsample} --save-exp-path "$save_exp_path" --save-exp-num "$save_exp_num" \
            --save-weights "$save_weights_path" --rewards-file "$save_rewards_path"
fi
