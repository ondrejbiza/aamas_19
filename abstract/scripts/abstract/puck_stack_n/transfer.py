import argparse
import itertools
import os
import copy as cp
import numpy as np
import tensorflow as tf
from ....envs.puck_stack import PuckStack
from ....abstraction.homomorphism_2 import Homomorphism
from ....models.branch_convnet import BranchConvNet
from ....agents import utils as agent_utils
from ....abstraction import utils as abstraction_utils
from ....models import utils as balancing_utils
from ....abstraction.graph import Graph
from ....agents.g import GGraphAgent
from ....agents.dqn_branch import DQNAgent
from ....models import models
from .... import vis_utils

BLOCK_SIZE = 28
ARCH = ((32, 64, 128), (3, 3, 3), (2, 2, 2), (32, 64), (128,))
LEARNING_RATE = 0.001
BATCH_SIZE = 32
WEIGHT_DECAY = 0.0001
MAX_REWARD = 10
MIN_SAMPLES = 5

DQN_LEARNING_RATE = 0.0001
DQN_BATCH_SIZE = 32
DQN_NUM_STEPS = 1000000
DQN_INITIAL_EPSILON = 1.0
DQN_TARGET_UPDATE_FREQ = 100


def run_task(args, previous_agent=None, previous_keys=None, previous_experience=None, previous_dqn=None):

    all_rewards = []
    dqn_exploration_fraction = args.dqn_num_exp_steps / DQN_NUM_STEPS

    if previous_keys is not None:
        keys = previous_keys
    else:
        keys = set()

    agent = None

    # treat previous experience
    if previous_experience is not None:
        for transition in previous_experience:
            transition.reward = 0

    # setup environment and helper functions
    env = PuckStack(num_pucks=args.num_pucks, num_blocks=args.num_blocks, min_radius=args.min_radius,
                    max_radius=args.max_radius, num_redundant_pucks=args.num_redundant_pucks)

    if previous_agent is not None:
        previous_agent.env = env

    num_positions = (args.num_blocks ** 2)
    num_actions = num_positions * 2

    # get initial behavior policy
    q_func = models.branch_cnn_to_mlp(
        convs=[(16, 3, 2), (32, 3, 2)],
        hand_hiddens=[16, 32],
        final_hiddens=[64]
    )

    if previous_dqn is None:
        dqn_agent = DQNAgent(env, 2, num_actions, q_func, tf.train.AdamOptimizer(learning_rate=DQN_LEARNING_RATE),
                             dqn_exploration_fraction, DQN_INITIAL_EPSILON, DQN_NUM_STEPS, args.dqn_final_epsilon,
                             DQN_BATCH_SIZE, target_update_freq=DQN_TARGET_UPDATE_FREQ, prioritized_replay=False,
                             custom_graph=True)
        dqn_agent.start_session(1, 0.1)
    else:
        dqn_agent = previous_dqn
        dqn_agent.env = env

        if args.share_dqn_reset_buffer:
            dqn_agent.empty_replay_buffer()

    def sample_actions(state):
        if state[1] == 0:
            start = 0
        else:
            start = num_positions
        end = start + num_positions
        return list(range(start, end))

    def get_block(state, action, reward):
        if reward > 0:
            return 1
        else:
            return 0

    def act(state, timestep):
        if timestep > 60:
            dqn_agent.learn(timestep)
        return dqn_agent.act(state, timestep)

    buffers = [[] for _ in range(2)]

    if previous_experience is not None:
        buffers[0] += previous_experience

    if args.num_episodes > 0:
        start_step_size = args.num_start_episodes // args.num_episodes
    else:
        start_step_size = 10

    num_act_steps = args.num_learn_act_steps + start_step_size

    while num_act_steps >= start_step_size:

        if previous_agent is not None:

            def option(state, timestep):
                return previous_agent.act(state, timestep)

            if args.no_remember:
                remember = None
            else:
                remember = dqn_agent.remember

            rewards, _ = abstraction_utils.collect_experience_option(
                env, act, get_block, buffers, option, max_buffer_size=args.max_buffer_size,
                num_episodes=args.num_start_episodes, verbose=args.verbose, keys=keys, task=args.task_idx,
                saver=args.saver, remember=remember
            )
        else:
            rewards, _ = abstraction_utils.collect_experience(
                env, act, get_block, buffers, max_buffer_size=args.max_buffer_size,
                num_episodes=args.num_start_episodes,
                verbose=args.verbose, keys=keys, task=args.task_idx, saver=args.saver
            )

        all_rewards.append(rewards)
        num_act_steps -= start_step_size

        if np.all([len(buffer) > MIN_SAMPLES for buffer in buffers]):
            break

    # delete previous agent
    if previous_agent is not None:
        previous_agent.model.reset()
        del previous_agent

    # join buffer into a single experience list
    experience = list(itertools.chain.from_iterable(buffers))

    # run the main loop
    for i in range(num_act_steps):

        # create a model for the g function and segment the MDP
        if args.reuse:
            fixed_num_classes = args.max_blocks
            assert fixed_num_classes is not None
        else:
            fixed_num_classes = None

        if args.undersample:
            def balance(x, y):
                return balancing_utils.oversample(*balancing_utils.undersample(x, y, target=args.undersample))
        else:
            balance = balancing_utils.oversample

        model = BranchConvNet([28 * args.num_blocks, 28 * args.num_blocks, 1], num_actions, 2, ARCH[0], ARCH[1],
                              ARCH[2], ARCH[3], ARCH[4], LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY,
                              max_training_steps=5000, set_best_params=True, preprocess=agent_utils.process_states,
                              balance=balance, fit_reset=not args.reuse,
                              early_stop_no_improvement_steps=args.early_stop, fixed_num_classes=fixed_num_classes,
                              verbose=args.verbose)

        homo = Homomorphism(
            experience, model, sample_actions, 10, state_action_threshold=args.state_action_threshold,
            state_action_confidence=args.state_action_confidence, threshold_reward_partition=False,
            nth_min_state_block_confidence=args.nth_min_confidence,
            state_block_confidence_product=args.confidence_product,
            max_blocks=args.max_blocks, reuse_blocks=args.reuse,
            adaptive_B_threshold=args.adaptive_state_action_threshold,
            adaptive_B_threshold_multiplier=args.adaptive_state_action_threshold_multiplier
        )

        homo.partition_iteration()

        # plan in the abstract MDP
        state_action_t = homo.partition.get_state_action_successor_table(majority_voting=True)

        state_t = homo.partition.get_state_successor_table()
        goal = homo.partition.get_goal_state_action_block(task_idx=args.task_idx)

        nodes = list(state_action_t.keys())
        edges = []

        for key, value in state_action_t.items():

            if value is None:
                continue

            next_blocks = state_t[value]
            for next_block in next_blocks:
                edges.append((key, next_block))

        graph = Graph(nodes, edges, goal, ignore_no_path=True)

        # use g to act in the environment
        agent = GGraphAgent(model, sample_actions, graph, env, softmax_selection=args.softmax_selection,
                            softmax_temperature=args.softmax_temperature, random_selection=args.random_selection,
                            alternative_policy=dqn_agent)

        if model.session is not None:
            def act(state, timestep):
                return agent.act(state, timestep)

        buffers = abstraction_utils.reclassify_experience(experience, get_block, 2)

        rewards, _ = abstraction_utils.collect_experience(
            env, act, get_block, buffers, num_episodes=args.num_episodes, max_buffer_size=args.max_buffer_size,
            verbose=args.verbose, keys=keys, task=args.task_idx, saver=args.saver
        )
        experience = list(itertools.chain.from_iterable(buffers))

        all_rewards.append(rewards)

        rewards = np.array(rewards)

        # maybe finish early
        if args.reward_threshold is not None and \
                (np.sum(rewards >= MAX_REWARD) / rewards.shape[0] >= args.reward_threshold):
            remaining_episodes = (args.num_learn_act_steps - (i + 1)) * args.num_episodes
            rewards, _ = abstraction_utils.collect_experience(
                env, act, get_block, buffers, num_episodes=remaining_episodes, max_buffer_size=args.max_buffer_size,
                verbose=args.verbose, keys=keys, task=args.task_idx, saver=args.saver
            )

            all_rewards.append(rewards)
            break

    if not args.share_dqn:
        # get rid of the DQN for the current task
        dqn_agent.stop_session()
        dqn_agent.delete_graph()

        if agent is not None:
            agent.alternative_policy = None

    return all_rewards, agent, experience, keys, dqn_agent


def main(args):

    # validate settings
    assert args.num_pucks_list is not None
    max_pucks = np.max(args.num_pucks_list)
    assert max_pucks <= args.num_blocks ** 2

    # limit gpu usage
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # maybe setup saver
    saver = None
    if args.gifs_dir is not None:
        saver = vis_utils.Saver(max_value=max_pucks, dir_path=args.gifs_dir)

    # run tasks
    all_rewards = []
    all_experience = None
    all_keys = set()
    last_agent = None
    last_dqn = None

    for task_idx, num_pucks in enumerate(args.num_pucks_list):

        # calculate number of redundant pucks so that there is always the same number of pucks in the environment
        num_redundant_pucks = max_pucks - num_pucks

        if args.num_redundant_pucks is not None:
            num_redundant_pucks += args.num_redundant_pucks

        # set arguments specific to the task
        task_args = cp.deepcopy(args)
        task_args.saver = saver
        task_args.num_pucks = num_pucks
        task_args.num_redundant_pucks = num_redundant_pucks
        task_args.task_idx = task_idx

        task_rewards, task_agent, task_experience, task_keys, task_dqn = run_task(
            task_args, previous_agent=last_agent, previous_keys=all_keys, previous_experience=all_experience,
            previous_dqn=last_dqn
        )

        if not args.no_option:
            last_agent = task_agent

        if args.share_dqn:
            last_dqn = task_dqn

        if not args.no_sharing:
            all_experience = task_experience
            all_keys.union(task_keys)

        all_rewards += task_rewards

    # save all rewards
    if args.rewards_file is not None:
        all_rewards = np.concatenate(all_rewards)
        np.savetxt(args.rewards_file, all_rewards)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run MDP minimization with branch convnet and random behavior policy.")

    parser.add_argument("num_blocks", type=int, help="number of blocks in the environment")
    parser.add_argument("num_learn_act_steps", type=int, help="number of segment-learn-act loops")
    parser.add_argument("--num-pucks-list", type=int, nargs="+", help="number of pucks to stack for each task")

    parser.add_argument("--num-start-episodes", type=int, default=100,
                        help="number of episodes to perform before minimization")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes to evaluate for")
    parser.add_argument("--max-buffer-size", type=int, default=10000)
    parser.add_argument("--reward-threshold", type=float,
                        help="if more than this fraction of episodes achieve max. reward don't split again")
    parser.add_argument("--no-option", default=False, action="store_true",
                        help="do not transfer agents between tasks")
    parser.add_argument("--no-sharing", default=False, action="store_true",
                        help="do not share experience between tasks")
    parser.add_argument("--no-remember", default=False, action="store_true",
                        help="do not remember experience generated by the option")
    parser.add_argument("--share-dqn", default=False, action="store_true",
                        help="share dqn weights")
    parser.add_argument("--share-dqn-reset-buffer", default=False, action="store_true",
                        help="reset the replay buffer of the shared DQN")

    parser.add_argument("--min-radius", type=int, default=7, help="minimum puck radius")
    parser.add_argument("--max-radius", type=int, default=12, help="maximum puck radius")
    parser.add_argument("--num-redundant-pucks", type=int, help="number of additional redundant pucks")

    # network settings
    parser.add_argument("--reuse", default=False, action="store_true", help="reuse previously trained blocks")
    parser.add_argument("--max-blocks", type=int, help="maximum state-action blocks; must be specified with reuse")
    parser.add_argument("--early-stop", type=int, help="stop training after this many episodes with no improvement")
    parser.add_argument("--undersample", type=int, help="maximum number of transitions per class")
    parser.add_argument("--dqn-num-exp-steps", type=int, default=2000)
    parser.add_argument("--dqn-final-epsilon", type=float, default=0.1)

    # homomorphism settings
    parser.add_argument("--state-action-threshold", type=int, default=50, help="state-action block split threshold")
    parser.add_argument("--state-action-confidence", type=float, default=0.0,
                        help="state-action block prediction confidence threshold")
    parser.add_argument("--nth-min-confidence", type=int, default=None, help="nth minimum confidence function")
    parser.add_argument("--confidence-product", default=False, action="store_true",
                        help="use confidence product function")
    parser.add_argument("--adaptive-state-action-threshold", default=False, action="store_true",
                        help="set state-action num. samples threshold best on classifier accuracy")
    parser.add_argument("--adaptive-state-action-threshold-multiplier", type=float, default=1.0,
                        help="the higher the multiplier the stricter the threshold")

    parser.add_argument("--softmax-selection", default=False, action="store_true", help="select action using softmax")
    parser.add_argument("--softmax-temperature", type=float, default=1.0, help="softmax action selection temperature")
    parser.add_argument("--random-selection", default=False, action="store_true", help="random block action selection")

    parser.add_argument("--rewards-file", help="where to save rewards")
    parser.add_argument("--gifs-dir", help="directory where to save gifs of all episodes")

    parser.add_argument("--verbose", default=False, action="store_true", help="print information")
    parser.add_argument("--gpus", help="comma-separated list of gpus to use")

    parsed = parser.parse_args()
    main(parsed)
