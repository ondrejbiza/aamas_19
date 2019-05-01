import argparse
import pickle
import numpy as np
from ....abstraction.homomorphism_3 import Homomorphism
from ....abstraction.quotient_mdp_deterministic import QuotientMDPFactory
from ....agents.quotient_mdp_agent import QuotientMDPAgent
from ....agents.solver import Solver
from ....agents import utils as agent_utils
from ....models.dilated_resnet_fc import DilatedResnetFC
from ....models import utils as balancing_utils
from ....envs.continuous_component import ContinuousComponent

NUM_ACTIONS = 112
NUM_BLOCKS = 4
NUM_EPISODES = 50

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


def main(args):

    # setup abstract model and the environment
    env = ContinuousComponent(
        num_pucks=args.num_pucks, num_blocks=NUM_BLOCKS, goal=args.num_pucks, num_actions=NUM_ACTIONS
    )

    # prepare experience
    with open(args.dataset_path, "rb") as file:
        experience = pickle.load(file)

    if args.limit is not None:
        experience = experience[-args.limit:]

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
        gpu_memory_fraction=args.gpu_memory_fraction, verbose=True, upsample_after=True,
        max_training_steps=args.drn_max_steps, set_best_params=True, preprocess=agent_utils.process_states,
        balance=balancing_utils.oversample, early_stop_no_improvement_steps=None,
        validation_fraction=validation_fraction, g_mean=args.g_mean, calibrate=args.calibrate,
        gauss_smooth=args.gauss_smooth, gauss_smooth_size=args.gauss_smooth_size,
        gauss_smooth_std=args.gauss_smooth_std, gauss_smooth_logits=args.gauss_smooth_logits,
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

    # evaluate the partition as an agent in the original environment
    quotient_mdp = QuotientMDPFactory.from_partition(homo.partition)
    quotient_mdp.value_iteration()
    q_values = quotient_mdp.get_state_action_block_q_values()

    agent = QuotientMDPAgent(
        model, q_values, env, fully_convolutional=True, proportional_selection=args.proportional_selection,
        softmax_selection=args.softmax_selection, softmax_temperature=args.softmax_temperature,
        random_selection=args.random_selection
    )

    solver = Solver(env, agent, NUM_EPISODES * 20, NUM_ACTIONS * 20, 0, max_episodes=NUM_EPISODES, train=False,
                    rewards_file=args.rewards_path)
    solver.run()

    print("rewards:", solver.episode_rewards)
    print("mean reward: {:.2f}".format(np.mean(solver.episode_rewards)))

    # maybe save partition report
    if args.report_path is not None:
        quotient_mdp.partition_report(homo.partition, model, args.report_path, num_actions=112, stride=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="path to a dataset of experience")
    parser.add_argument("num_pucks", type=int, help="number of pucks to stack")
    parser.add_argument("threshold", type=int, help="state-action block size threshold")
    parser.add_argument("max_blocks", type=int, help="maximum number of state-action blocks")

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
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--drn-cap", type=int, default=32)
    parser.add_argument("--drn-lr-steps", type=int, nargs="+", default=LEARNING_RATE_STEPS)
    parser.add_argument("--drn-lr-values", type=float, nargs="+", default=LEARNING_RATE_VALUES)
    parser.add_argument("--drn-max-steps", type=int, default=NUM_TRAINING_STEPS)

    parser.add_argument("--gauss-smooth", default=False, action="store_true")
    parser.add_argument("--gauss-smooth-size", type=int, default=5)
    parser.add_argument("--gauss-smooth-std", type=float, default=1.0)
    parser.add_argument("--gauss-smooth-logits", default=False, action="store_true")

    parser.add_argument("--proportional-selection", default=False, action="store_true")
    parser.add_argument("--softmax-selection", default=False, action="store_true")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--random-selection", default=False, action="store_true")

    parser.add_argument("--rewards-path", default=None)
    parser.add_argument("--report-path", default=None)

    parser.add_argument("--gpu-memory-fraction", type=float, default=None)

    parsed = parser.parse_args()
    main(parsed)
