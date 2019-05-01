import argparse
import pickle
from ....abstraction.homomorphism_3 import Homomorphism
from ....abstraction.quotient_mdp_deterministic import QuotientMDPFactory
from ....agents import utils as agent_utils
from ....agents.dqn_fc import DQNFC
from ....agents.options_agent import OptionsAgent
from ....agents.quotient_mdp_agent import QuotientMDPAgent
from ....agents.solver import Solver
from ....agents.transfer_agent import TransferAgent
from ....models.dilated_resnet_fc import DilatedResnetFC
from ....models import utils as balancing_utils
from .... import constants
from ....envs.continuous_two_stacks import ContinuousTwoStacks

# environment settings
NUM_ACTIONS = 112
NUM_BLOCKS = 4
BLOCK_SIZE = 28

# partitioning settings
PRE_CONV = ((16,), (7,), (1,), (1,))

BLOCKS_CAP_32 = ((16, 32, 32, 32, 32, 32, 32, 32, 32, 32),
          (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
          (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
POST_CONV_CAP_32 = ((32, 32, 32, 32), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

BLOCKS_CAP_64 = ((16, 32, 64, 64, 64, 64, 64, 64, 64, 64),
          (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
          (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
POST_CONV_CAP_64 = ((64, 64, 64, 64), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

BLOCKS_CAP_128 = ((16, 32, 64, 64, 128, 128, 128, 128, 128, 128),
          (1, 2, 2, 1, 1, 1, 1, 1, 1, 1),
          (1, 1, 1, 1, 1, 1, 2, 2, 4, 4))
POST_CONV_CAP_128 = ((128, 128, 128, 128), (3, 3, 3, 3), (1, 1, 1, 1), (2, 2, 1, 1))

BATCH_NORM = True
LEARNING_RATE_STEPS = [500, 1000]
LEARNING_RATE_VALUES = [0.1, 0.01, 0.001]
NUM_TRAINING_STEPS = 1500
BATCH_SIZE = 64
WEIGHT_DECAY = 0.0001
VALIDATION_FRACTION = 0.2

# DQN settings
LEARNING_STARTS = 60
TRAIN_FREQ = 1
PRINT_FREQ = 1


def main(args):

    # prepare experience
    with open(args.dataset_path, "rb") as file:
        experience = pickle.load(file)

    for transition in experience:
        transition.action = transition.action % (NUM_ACTIONS ** 2)

    # create a model and run online partition iteration
    if args.no_validation:
        validation_fraction = None
    else:
        validation_fraction = VALIDATION_FRACTION

    if args.drn_cap == 32:
        blocks = BLOCKS_CAP_32
        post_conv = POST_CONV_CAP_32
    elif args.drn_cap == 64:
        blocks = BLOCKS_CAP_64
        post_conv = POST_CONV_CAP_64
    elif args.drn_cap == 128:
        blocks = BLOCKS_CAP_128
        post_conv = POST_CONV_CAP_128
    else:
        raise NotImplementedError("This build is not implemented.")

    model = DilatedResnetFC(
        [112, 112, 1], [NUM_ACTIONS, NUM_ACTIONS], PRE_CONV[0], PRE_CONV[1], PRE_CONV[2], PRE_CONV[3],
        blocks[0], blocks[1], blocks[2], post_conv[0], post_conv[1], post_conv[2], post_conv[3],
        gpu_memory_fraction=args.homo_gpu_memory_fraction, verbose=True, upsample_after=True,
        max_training_steps=args.drn_max_steps, set_best_params=True, preprocess=agent_utils.process_states,
        balance=balancing_utils.oversample, early_stop_no_improvement_steps=None,
        validation_fraction=validation_fraction, g_mean=args.g_mean, calibrate=args.calibrate,
        batch_norm=BATCH_NORM, batch_size=BATCH_SIZE, valid_batch_size=BATCH_SIZE,
        learning_rate_steps=args.drn_lr_steps, learning_rate_values=args.drn_lr_values
    )

    homo = Homomorphism(
        experience, model, args.max_blocks, sample_actions=None, fully_convolutional=True,
        size_threshold=args.threshold, confidence_threshold=args.confidence_threshold, show_average_confidences=True,
        confidence_percentile=args.confidence_percentile, deduplicate_before_training=args.deduplicate,
        ignore_low_conf=args.ignore_low_conf, exclude_blocks=args.exclude_blocks, verbose=True,
        confidence_propagation=args.confidence_propagation, reuse_network=False
    )

    homo.partition_iteration()

    # create an environment for the second task and a DQN
    assert args.upsample in [constants.UPSAMPLE_BEFORE, constants.UPSAMPLE_AFTER, constants.DECONV_BEFORE,
                             constants.DECONV_AFTER]
    upsample_before = args.upsample == constants.UPSAMPLE_BEFORE
    upsample_after = args.upsample == constants.UPSAMPLE_AFTER
    deconv_before = args.upsample == constants.DECONV_BEFORE
    deconv_after = args.upsample == constants.DECONV_AFTER

    # setup environment
    env = ContinuousTwoStacks(NUM_BLOCKS, args.num_pucks, args.num_pucks // 2, NUM_ACTIONS)
    env.initStride = args.init_env_stride  # stride for initial puck placement
    env.stride = args.env_stride  # stride for action specification

    state_shape = (NUM_BLOCKS * BLOCK_SIZE, NUM_BLOCKS * BLOCK_SIZE, 1)
    output_shape = (NUM_ACTIONS, NUM_ACTIONS)

    dqn_agent = DQNFC(
        env, state_shape, output_shape, args.num_filters, args.filter_sizes, args.strides, args.learning_rate,
        args.batch_size, constants.OPT_MOMENTUM, args.exploration_fraction, 1.0, args.final_epsilon,
        args.max_time_steps, buffer_size=args.buffer_size, prioritized_replay=not args.disable_prioritized_replay,
        target_net=not args.disable_target_network, target_update_freq=args.target_update_freq,
        show_q_values=args.show_q_values, show_q_values_offset=args.show_q_values_offset,
        upsample_before=upsample_before, upsample_after=upsample_after, deconv_before=deconv_before,
        deconv_after=deconv_after, deconv_before_num_filters=not args.deconv_before_num_filters,
        deconv_filter_size=args.deconv_filter_size, end_filter_sizes=args.end_filter_sizes
    )

    dqn_agent.start_session(2, args.dqn_gpu_memory_fraction)

    # create an options agent
    quotient_mdp = QuotientMDPFactory.from_partition(homo.partition)
    quotient_mdp.value_iteration()
    q_values = quotient_mdp.get_state_action_block_q_values()

    quotient_mdp_agent = QuotientMDPAgent(
        model, q_values, env, fully_convolutional=True, proportional_selection=args.proportional_selection,
        softmax_selection=args.softmax_selection, softmax_temperature=args.softmax_temperature,
        random_selection=args.random_selection
    )

    options_agent = OptionsAgent(
        quotient_mdp, quotient_mdp_agent, OptionsAgent.Exploration.EPSILON_GREEDY,
        learning_rate=args.options_learning_rate, epsilon=args.options_epsilon, discount=args.options_discount
    )

    # create a transfer agent
    transfer_agent = TransferAgent(options_agent, dqn_agent, dqn_remember_option=True)

    # run the transfer agent
    solver = Solver(
        env, transfer_agent, args.max_time_steps, learning_start=LEARNING_STARTS, train_freq=TRAIN_FREQ,
        max_episodes=args.max_episodes, rewards_file=args.rewards_file,
        animate=args.animate, animate_from=args.animate_from,
        gif_save_path=args.save_gifs_path, gif_save_limit=args.save_limit,
        gif_save_only_successful=args.save_only_successful, max_depth_value=args.num_pucks
    )

    # solve the environment
    solver.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument("dataset_path", help="path to a dataset of experience for the first task")
    parser.add_argument("num_pucks", type=int, help="number of pucks for the second task")
    parser.add_argument("threshold", type=int, help="state-action block size threshold")
    parser.add_argument("max_blocks", type=int, help="maximum number of state-action blocks")

    # partitioning settings
    parser.add_argument("--state-action-confidence-histogram", default=False, action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.)
    parser.add_argument("--confidence-percentile", type=int)
    parser.add_argument("--deduplicate", default=False, action="store_true")
    parser.add_argument("--ignore-low-conf", default=False, action="store_true")
    parser.add_argument("--exclude-blocks", default=False, action="store_true")
    parser.add_argument("--calibrate", default=False, action="store_true")
    parser.add_argument("--normalize", default=False, action="store_true")
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--no-validation", default=False, action="store_true")
    parser.add_argument("--confidence-propagation", default=False, action="store_true",
                        help="propagate confidence about the dataset")
    parser.add_argument("--g-mean", default=False, action="store_true",
                        help="use geometric mean instead of arithmetic mean to combine per-class accuracies")

    parser.add_argument("--drn-cap", type=int, default=32)
    parser.add_argument("--drn-lr-steps", type=int, nargs="+", default=LEARNING_RATE_STEPS)
    parser.add_argument("--drn-lr-values", type=float, nargs="+", default=LEARNING_RATE_VALUES)
    parser.add_argument("--drn-max-steps", type=int, default=NUM_TRAINING_STEPS)

    # options settings
    parser.add_argument("--proportional-selection", default=False, action="store_true")
    parser.add_argument("--softmax-selection", default=False, action="store_true")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--random-selection", default=False, action="store_true")

    parser.add_argument("--options-learning-rate", type=float, default=0.1)
    parser.add_argument("--options-epsilon", type=float, default=0.1)
    parser.add_argument("--options-discount", type=float, default=0.9)

    # DQN settings
    parser.add_argument("--num-filters", type=int, nargs="+", default=[32, 64, 64, 32])
    parser.add_argument("--filter-sizes", type=int, nargs="+", default=[8, 8, 3, 1])
    parser.add_argument("--strides", type=int, nargs="+", default=[4, 2, 1, 1])
    parser.add_argument("--upsample", default=constants.UPSAMPLE_AFTER,
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
    parser.add_argument("--buffer-size", type=int, default=100000)

    # environment setting
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

    parser.add_argument("--homo-gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--dqn-gpu-memory-fraction", type=float, default=None)

    parsed = parser.parse_args()
    main(parsed)
