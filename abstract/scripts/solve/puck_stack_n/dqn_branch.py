import argparse
import os
import pickle
import tensorflow as tf
from ....agents.dqn_branch import DQNAgent
from ....agents.solver import Solver
from ....models import utils as balancing_utils
from ....models import models
from ....envs.puck_stack import PuckStack

LEARNING_STARTS = 60
BUFFER_SIZE = 100000
TRAIN_FREQ = 1
PRINT_FREQ = 1


def main(args):

    # setup environment
    env = PuckStack(num_pucks=args.num_pucks, num_blocks=args.num_blocks)
    env.initStride = args.init_env_stride  # stride for initial puck placement
    env.stride = args.env_stride  # stride for action specification

    num_states = 2  # either holding or not
    num_patches = len(env.moveCenters) ** 2
    num_actions = 2 * num_patches

    # create neural network
    q_func = models.branch_cnn_to_mlp(
        convs=[(16, 3, 2), (32, 3, 2)],
        hand_hiddens=[16, 32],
        final_hiddens=[64]
    )

    agent = DQNAgent(env, num_states, num_actions, q_func, tf.train.AdamOptimizer(learning_rate=args.learning_rate),
                     args.exploration_fraction, 1.0, args.max_time_steps,args.final_epsilon, args.batch_size,
                     buffer_size=BUFFER_SIZE, prioritized_replay=False, target_network=True, target_update_freq=100)

    agent.start_session(args.num_cpu, args.gpu_memory_fraction)

    # initialize a solver
    solver = Solver(env, agent, args.max_time_steps, learning_start=LEARNING_STARTS, train_freq=TRAIN_FREQ,
                    max_episodes=args.max_episodes, rewards_file=args.rewards_file,
                    abstract_actions_file=args.abstract_actions_file, animate=args.animate,
                    animate_from=args.animate_from, gif_save_path=args.save_gifs_path, gif_save_limit=args.save_limit,
                    gif_save_only_successful=args.save_only_successful, max_depth_value=args.num_pucks)

    # solve the environment
    solver.run()

    # stop session
    agent.stop_session()

    # maybe save the collected experience
    if args.save_exp_path is not None:

        transitions = balancing_utils.get_experience_from_replay_buffer(agent.replay_buffer, limit=args.save_exp_num)

        save_dir = os.path.dirname(args.save_exp_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        with open(args.save_exp_path, "wb") as file:
            pickle.dump(transitions, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Solve stacking pucks with the DQN agent.")

    parser.add_argument("num_pucks", type=int, help="number of pucks in the environment")
    parser.add_argument("num_blocks", type=int, help="size of the environment")

    parser.add_argument("--learning-rate", type=float, default=0.0001, help="learning rate for gamma neural networks")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for the learning of gamma")
    parser.add_argument("--exploration-fraction", type=float, default=1.0,
                        help="how many time steps to explore expressed as the faction of the maximum number of "
                             "time steps")
    parser.add_argument("--final-epsilon", type=float, default=0.1,
                        help="value of epsilon after the end of exploration")

    parser.add_argument("--init-env-stride", type=int, default=28,
                        help="stride for the placement of objects in the environment")
    parser.add_argument("--env-stride", type=int, default=28, help="stride for the actions")
    parser.add_argument("--max-time-steps", type=int, default=2000, help="maximum number of time steps to run for")
    parser.add_argument("--max-episodes", type=int, default=None, help="maximum number of episodes to run for")
    parser.add_argument("--rewards-file", default=None, help="where to save the per-episode rewards")
    parser.add_argument("--abstract-actions-file", help="where to save the per-step abstract actions")

    parser.add_argument("--animate", default=False, action="store_true", help="show an animation of the environment")
    parser.add_argument("--animate-from", type=int, default=0, help="from which episode to start the animation")

    parser.add_argument("--save-gifs-path", help="save path for gifs of episodes")
    parser.add_argument("--save-only-successful", default=False, action="store_true",
                        help="save only the successful episodes")
    parser.add_argument("--save-limit", type=int, help="maximum number of episodes to save")

    parser.add_argument("--gpu-memory-fraction", type=float, default=0.1,
                        help="a fraction of GPU memory to use; None for all")
    parser.add_argument("--num-cpu", type=int, default=1, help="number of CPUs to use")

    parser.add_argument("--save-exp-path", help="where to save the collected experience")
    parser.add_argument("--save-exp-num", type=int, default=20000, help="number of transitions to save")

    parsed = parser.parse_args()
    main(parsed)
