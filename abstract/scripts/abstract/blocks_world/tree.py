import argparse
import numpy as np
from .... import constants
from ....envs.blocks_world import BlocksWorldEnv
from ....abstraction.homomorphism_2 import Homomorphism
from ....models.d_tree import DTree
from ....abstraction import utils as abstraction_utils
from ....abstraction.graph import Graph
from ....agents.g import GGraphAgent

MIN_SAMPLES = 0
BASELINE_MAX_REWARD = 22.368


def run_task(args):

    # maybe filter experience from previous tasks
    if args.transfer_only_rewards:
        if args.experience_dict is not None:

            to_delete = []
            task_counts = {}

            for key, value in args.experience_dict.items():

                if value.task not in task_counts:
                    task_counts[value.task] = 0

                if value.reward > 0:
                    task_counts[value.task] += 1

                if (value.reward <= 0 and value.task != args.task_idx) or \
                        (args.rewards_transfer_limit is not None and task_counts[value.task] > args.rewards_transfer_limit):
                    to_delete.append(key)

            for key in to_delete:
                del args.experience_dict[key]

    # setup environment and helper functions
    if args.relational:
        encode_all = False
        encode_relational = True
    else:
        encode_all = True
        encode_relational = False

    env = BlocksWorldEnv(encode_all=encode_all, encode_relational=encode_relational, include_colors=args.include_colors)

    if args.task is not None:
        env.target_position = args.task[0]
        env.target_height = args.task[1]
    else:
        env.reset_goal(blacklist=args.blacklist)
        args.blacklist.append((env.target_position, env.target_height))

    num_actions = env.NUM_ACTIONS

    # get initial behavior policy
    def sample_actions(state):
        return list(range(0, num_actions))

    def act(state, timestep):
        actions = sample_actions(state)
        action = np.random.choice(actions)
        next_state, reward, done, _ = env.step(action)
        return None, None, None, action, next_state, None, None, None, reward, done

    # collect random experience
    num_act_steps = args.num_learn_act_steps + 1
    global_timestep = 0

    while num_act_steps > 0:
        rewards, num_timesteps, num_rewards = abstraction_utils.collect_experience_dictionary(
            env, act, args.experience_dict, num_episodes=args.num_start_episodes,
            num_timesteps=args.num_start_timesteps, task=args.task_idx
        )

        args.all_rewards.append(rewards)
        args.all_mean_rewards.append(np.sum(rewards) / num_timesteps)
        num_act_steps -= 1

        if num_rewards > MIN_SAMPLES:
            break

    # join buffer into a single experience list
    experience = list(args.experience_dict.values())

    # run the main loop
    for i in range(num_act_steps):

        # create a model for the g function and segment the MDP
        model = DTree()

        homo = Homomorphism(
            experience, model, sample_actions, 10, state_action_threshold=args.state_action_threshold,
            state_action_confidence=args.state_action_confidence, threshold_reward_partition=False,
            nth_min_state_block_confidence=args.nth_min_confidence,
            state_block_confidence_product=args.confidence_product,
            max_blocks=args.max_blocks
        )

        homo.partition_iteration()

        if args.max_buffer_size is not None:
            experience = []

            for block in homo.partition.blocks:
                np.random.shuffle(block)
                experience += block[:args.max_buffer_size]

            args.experience_dict = {}
            for transition in experience:
                args.experience_dict[tuple(transition.state) + (transition.action,)] = transition

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
                            confidence_exp=args.confidence_exp, exp_timesteps=args.exp_timesteps,
                            exp_softmax=args.exp_softmax)

        if model.trained:
            def act(state, timestep):
                return agent.act(state, timestep)

        prev_global_timestep = global_timestep

        rewards, global_timestep, _ = abstraction_utils.collect_experience_dictionary(
            env, act, args.experience_dict, num_episodes=args.num_episodes, num_timesteps=args.num_timesteps,
            start_timestep=global_timestep, task=args.task_idx
        )

        experience = list(args.experience_dict.values())

        args.all_rewards.append(rewards)

        rewards = np.array(rewards)
        args.all_mean_rewards.append(np.sum(rewards) / (global_timestep - prev_global_timestep))

        print("|B|: {:d}".format(len(homo.partition.blocks)))
        print("|S/B|: {:d}".format(len(homo.partition.state_keys)))
        print("mean reward per timesteps: {:.2f}".format(args.all_mean_rewards[-1]))

        # maybe finish early
        if args.mean_reward_threshold is not None and args.all_mean_rewards[-1] >= args.mean_reward_threshold:

            prev_global_timestep = global_timestep

            if args.num_episodes is not None:
                remaining_episodes = (num_act_steps - (i + 1)) * args.num_episodes
            else:
                remaining_episodes = None

            if args.num_timesteps is not None:
                remaining_timesteps = (num_act_steps - (i + 1)) * args.num_timesteps
            else:
                remaining_timesteps = None

            rewards, global_timestep, _ = abstraction_utils.collect_experience_dictionary(
                env, act, args.experience_dict, num_episodes=remaining_episodes, num_timesteps=remaining_timesteps,
                start_timestep=global_timestep, task=args.task_idx
            )

            args.all_rewards.append(rewards)

            for _ in range(args.num_learn_act_steps - (i + 1)):
                args.all_mean_rewards.append(np.sum(rewards) / (global_timestep - prev_global_timestep))

            print("mean reward per timesteps: {:.2f}".format(args.all_mean_rewards[-1]))

            model.reset()
            break

        # free memory
        model.reset()

    print(args.all_mean_rewards)
    print("all: ", np.mean(args.all_mean_rewards))


def main(args):

    # validate parameters
    assert args.difficulty in [constants.EASY, constants.MEDIUM, constants.HARD]
    assert args.tasks is None or (args.num_tasks * 2 == len(args.tasks))
    assert (args.num_episodes is not None or args.num_timesteps is not None) \
        and args.num_episodes != args.num_timesteps
    assert (args.num_start_episodes is not None or args.num_start_timesteps is not None) \
        and args.num_start_episodes != args.num_start_timesteps

    # prepare containers for results
    args.blacklist = []
    args.all_rewards = []
    args.all_mean_rewards = []
    args.experience_dict = {}

    # run tasks
    for task_idx in range(args.num_tasks):

        if args.no_transfer:
            args.experience_dict = {}

        if args.tasks is not None:
            args.task = args.tasks[task_idx * 2 : (task_idx + 1) * 2]
        else:
            args.task = None

        args.task_idx = task_idx
        run_task(args)

    # save all rewards
    if args.rewards_file is not None:
        np.savetxt(args.rewards_file, np.concatenate(args.all_rewards))

    # save all mean rewards
    if args.mean_rewards_file is not None:
        np.savetxt(args.mean_rewards_file, args.all_mean_rewards)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run MDP minimization with a decision tree and a random behavior policy.")

    parser.add_argument("num_learn_act_steps", type=int, help="number of segment-learn-act loops")
    parser.add_argument("num_tasks", type=int, help="number of tasks to solve")

    # environment settings
    parser.add_argument("--no-transfer", default=False, action="store_true", help="disable block transfer")
    parser.add_argument("--transfer-only-rewards", default=False, action="store_true",
                        help="transfer only rewards from previous tasks")
    parser.add_argument("--rewards-transfer-limit", type=int,
                        help="number of rewards that will be kept for each task")
    parser.add_argument("--include-colors", default=False, action="store_true",
                        help="include block color as a confounding variable")
    parser.add_argument("--difficulty", default=constants.EASY,
                        help="{}, {} or {}".format(constants.EASY, constants.MEDIUM, constants.HARD))

    parser.add_argument("--max-buffer-size", type=int, help="maximum number of experience for each class")
    parser.add_argument("--mean-reward-threshold", type=float,
                        help="if more than this fraction of episodes achieve max. reward don't split again")
    parser.add_argument("--relational", default=False, action="store_true",
                        help="include relational features as described in Wolfe's paper")

    parser.add_argument("--num-timesteps", type=int, help="number of timesteps")
    parser.add_argument("--num-start-timesteps", type=int, help="number of timesteps to perform before minimization")
    parser.add_argument("--num-episodes", type=int, help="number of episodes to evaluate for")
    parser.add_argument("--num-start-episodes", type=int, help="number of episodes to perform before minimization")

    parser.add_argument("--tasks", nargs="+", type=int, help="tuples for each task (target position, target height)")
    parser.add_argument("--max-blocks", type=int, help="maximum state-action blocks; must be specified with reuse")

    # homomorphism settings
    parser.add_argument("--state-action-threshold", type=int, default=50, help="state-action block split threshold")
    parser.add_argument("--state-action-confidence", type=float, default=0.0,
                        help="state-action block prediction confidence threshold")
    parser.add_argument("--nth-min-confidence", type=int, default=None, help="nth minimum confidence function")
    parser.add_argument("--confidence-product", default=False, action="store_true",
                        help="use confidence product function")

    # action selection settings
    parser.add_argument("--softmax-selection", default=False, action="store_true", help="select action using softmax")
    parser.add_argument("--softmax-temperature", type=float, default=1.0, help="softmax action selection temperature")
    parser.add_argument("--random-selection", default=False, action="store_true", help="random block action selection")

    parser.add_argument("--confidence-exp", default=False, action="store_true", help="confidence-based exploration")
    parser.add_argument("--exp-timesteps", type=int, help="for how many timesteps to explore")
    parser.add_argument("--exp-softmax", default=False, action="store_true", help="use softmax when sampling actions")

    parser.add_argument("--rewards-file", help="where to save rewards")
    parser.add_argument("--mean-rewards-file", help="where to save mean rewards per timestep")

    parser.add_argument("--verbose", default=False, action="store_true", help="print information")

    parsed = parser.parse_args()
    main(parsed)
