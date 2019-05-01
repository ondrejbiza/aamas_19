import argparse
import os
import pickle
from ....agents.dqn_fc import DQNFC
from ....agents.solver import Solver
from ....models import utils as balancing_utils
from ....envs.continuous_stairs import ContinuousStairs
from .... import constants

LEARNING_STARTS = 60
TRAIN_FREQ = 1
PRINT_FREQ = 1


def main(args):

    # process arguments
    assert args.upsample in [constants.UPSAMPLE_BEFORE, constants.UPSAMPLE_AFTER, constants.DECONV_BEFORE,
                             constants.DECONV_AFTER]
    upsample_before = args.upsample == constants.UPSAMPLE_BEFORE
    upsample_after = args.upsample == constants.UPSAMPLE_AFTER
    deconv_before = args.upsample == constants.DECONV_BEFORE
    deconv_after = args.upsample == constants.DECONV_AFTER

    # setup environment
    env = ContinuousStairs(
        args.num_blocks, args.num_pucks + args.num_redundant_pucks, args.num_pucks, args.num_actions
    )
    env.initStride = args.init_env_stride  # stride for initial puck placement
    env.stride = args.env_stride  # stride for action specification

    # setup the agent
    state_shape = (args.num_blocks * 28, args.num_blocks * 28, 1)
    output_shape = (args.num_actions, args.num_actions)

    agent = DQNFC(
        env, state_shape, output_shape, args.num_filters, args.filter_sizes, args.strides, args.learning_rate,
        args.batch_size, constants.OPT_MOMENTUM, args.exploration_fraction, 1.0, args.final_epsilon,
        args.max_time_steps, buffer_size=args.buffer_size, prioritized_replay=not args.disable_prioritized_replay,
        target_net=not args.disable_target_network, target_update_freq=args.target_update_freq,
        show_q_values=args.show_q_values, show_q_values_offset=args.show_q_values_offset,
        upsample_before=upsample_before, upsample_after=upsample_after, deconv_before=deconv_before,
        deconv_after=deconv_after, deconv_before_num_filters=not args.deconv_before_num_filters,
        deconv_filter_size=args.deconv_filter_size, end_filter_sizes=args.end_filter_sizes,
        dilation=args.dilation, double_q_network=args.double_q_network, use_memory=args.memory
    )

    agent.start_session(args.num_cpu, args.gpu_memory_fraction)

    # maybe load weights
    if args.load_weights:
        agent.load(args.load_weights)
        print("Loaded weights.")

    # initialize a solver
    solver = Solver(
        env, agent, args.max_time_steps, learning_start=LEARNING_STARTS, train_freq=TRAIN_FREQ,
        max_episodes=args.max_episodes, rewards_file=args.rewards_file,
        animate=args.animate, animate_from=args.animate_from,
        gif_save_path=args.save_gifs_path, gif_save_limit=args.save_limit,
        gif_save_only_successful=args.save_only_successful, max_depth_value=args.num_pucks
    )

    # solve the environment
    solver.run()

    # save the weights of the network
    if args.save_weights is not None:
        agent.save(args.save_weights)

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

    parser = argparse.ArgumentParser("Learn to build stairs with a fully-convolutional DQN agent.")

    parser.add_argument("num_blocks", type=int, default=4, help="number of blocks in the environment")
    parser.add_argument("num_actions", type=int, default=112, help="number of actions")
    parser.add_argument("num_pucks", type=int, default=2, help="number of pucks in the environment")

    parser.add_argument("--num-filters", type=int, nargs="+", default=[32, 64, 64, 32])
    parser.add_argument("--filter-sizes", type=int, nargs="+", default=[8, 8, 3, 1])
    parser.add_argument("--strides", type=int, nargs="+", default=[4, 2, 1, 1])
    parser.add_argument("--dilation", type=int, nargs="+", default=None)
    parser.add_argument("--upsample", default=constants.DECONV_AFTER,
                        help="{}, {}, {}, {}".format(
                            constants.UPSAMPLE_BEFORE, constants.UPSAMPLE_AFTER, constants.DECONV_BEFORE,
                            constants.DECONV_AFTER
                        ))
    parser.add_argument("--deconv-before-num-filters", type=int, default=32)
    parser.add_argument("--deconv-filter-size", type=int, default=16)
    parser.add_argument("--end-filter-sizes", type=int, default=None, nargs="+")

    parser.add_argument("--learning-rate", type=float, default=0.0001, help="learning rate for gamma neural networks")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for the learning of gamma")
    parser.add_argument("--target-update-freq", type=int, default=100, help="update frequency of the target network")
    parser.add_argument("--exploration-fraction", type=float, default=1.0,
                        help="how many time steps to explore expressed as the faction of the maximum number of "
                             "time steps")
    parser.add_argument("--final-epsilon", type=float, default=0.1,
                        help="value of epsilon after the end of exploration")
    parser.add_argument("--disable-prioritized-replay", default=False, action="store_true",
                        help="disable prioritized replay")
    parser.add_argument("--disable-target-network", default=False, action="store_true", help="disable target DQN")
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--double-q-network", default=False, action="store_true")
    parser.add_argument("--memory", default=False, action="store_true")

    parser.add_argument("--num-redundant-pucks", type=int, default=0)
    parser.add_argument("--init-env-stride", type=int, default=28,
                        help="stride for the placement of objects in the environment")
    parser.add_argument("--env-stride", type=int, default=28, help="stride for the actions")
    parser.add_argument("--max-time-steps", type=int, default=2000, help="maximum number of time steps to run for")
    parser.add_argument("--max-episodes", type=int, default=None, help="maximum number of episodes to run for")
    parser.add_argument("--rewards-file", default=None, help="where to save the per-episode rewards")

    parser.add_argument("--animate", default=False, action="store_true", help="show an animation of the environment")
    parser.add_argument("--animate-from", type=int, default=0, help="from which episode to start the animation")
    parser.add_argument("--show-q-values", default=False, action="store_true")
    parser.add_argument("--show-q-values-offset", type=int, default=False,
                        help="offset in terms of the number of timesteps")

    parser.add_argument("--save-gifs-path", help="save path for gifs of episodes")
    parser.add_argument("--save-only-successful", default=False, action="store_true",
                        help="save only the successful episodes")
    parser.add_argument("--save-limit", type=int, help="maximum number of episodes to save")

    parser.add_argument("--gpu-memory-fraction", type=float, default=0.1,
                        help="a fraction of GPU memory to use; None for all")
    parser.add_argument("--num-cpu", type=int, default=1, help="number of CPUs to use")

    parser.add_argument("--save-exp-path", help="where to save the collected experience")
    parser.add_argument("--save-exp-num", type=int, default=20000, help="number of transitions to save")

    parser.add_argument("--save-weights", help="where to save the weights of the network")
    parser.add_argument("--load-weights", help="load weights of the network")

    parsed = parser.parse_args()
    main(parsed)
