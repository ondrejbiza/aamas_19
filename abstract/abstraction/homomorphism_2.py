import collections
import numpy as np
from .partition import StateActionPartition
from . import utils


class Homomorphism:

    def __init__(self, experience, g, sample_actions, partition_iteration_max_steps, state_action_threshold=0,
                 state_threshold=0, state_action_confidence=0.0, state_confidence=0.0,
                 nth_min_state_block_confidence=None, state_block_confidence_product=False,
                 threshold_reward_partition=True, visualize_state_action_partition=None, visualize_state_partition=None,
                 visualize_split=False, max_blocks=None, reuse_blocks=False, reset_logits=False, orphan_check=False,
                 save_state_action_partition=None, save_state_partition=None, save_all=False, save_path=None,
                 adaptive_B_threshold=False, adaptive_B_threshold_multiplier=1.0, verbose=False,
                 confidence_percentile=None, ignore_low_conf=False, show_average_confidences=False,
                 exclude_blocks=False, deduplicate_before_training=False, fully_convolutional=False,
                 confidence_propagation=False):
        """
        MDP minimization algorithm.
        :param experience:                          List of transitions.
        :param g:                                   State-action block classifier.
        :param sample_actions:                      Sample actions for a given state.
        :param partition_iteration_max_steps:       Maximum number of steps of partition iteration.
        :param threshold_reward_partition:          Apply state_action_slit_threshold to partitions with reward != 0
                                                    (assuming reward = 0 is the most common).
        :param orphan_check:                        Check for orphaned state blocks.
        """

        assert not reuse_blocks or max_blocks is not None

        self.experience = experience
        self.g = g
        self.sample_actions = sample_actions

        self.state_action_threshold = state_action_threshold
        self.state_threshold = state_threshold
        self.state_action_confidence = state_action_confidence
        self.state_confidence = state_confidence
        self.nth_min_state_block_confidence = nth_min_state_block_confidence
        self.state_block_confidence_product = state_block_confidence_product
        self.adaptive_B_threshold = adaptive_B_threshold
        self.adaptive_B_threshold_multiplier = adaptive_B_threshold_multiplier
        self.confidence_percentile = confidence_percentile
        self.ignore_low_conf = ignore_low_conf
        self.exclude_blocks = exclude_blocks

        self.partition_iteration_max_steps = partition_iteration_max_steps
        self.threshold_reward_partition = threshold_reward_partition
        self.visualize_state_action_partition = visualize_state_action_partition
        self.visualize_state_partition = visualize_state_partition
        self.visualize_split = visualize_split
        self.max_blocks = max_blocks
        self.reuse_blocks = reuse_blocks
        self.reset_logits = reset_logits
        self.orphan_check = orphan_check
        self.save_state_action_partition = save_state_action_partition
        self.save_state_partition = save_state_partition
        self.save_all = save_all
        self.save_path = save_path
        self.verbose = verbose
        self.show_average_confidences = show_average_confidences
        self.deduplicate_before_training = deduplicate_before_training
        self.fully_convolutional = fully_convolutional
        self.confidence_propagation = confidence_propagation

        self.partition = StateActionPartition()
        self.partition.add(experience)

        self.first_step = True
        self.state_action_save_count = 0
        self.state_save_count = 0

        self.block_confs = None

    def partition_iteration(self):
        """
        Run partition iteration.
        :return:        Number of steps before convergence (or before the limit was reached).
        """

        step = 0
        change = True

        while change and step < self.partition_iteration_max_steps:
            change = self.partition_improvement()
            step += 1

            # early termination
            if self.max_blocks is not None and len(self.partition.blocks) >= self.max_blocks:
                return step

        return step

    def partition_improvement(self):
        """
        Run a single iteration of partition improvement.
        :return:        True if the partition changed, otherwise False.
        """

        if self.first_step:
            # first create a reward-respecting partition
            some_change = self.split_rewards()
            self.first_step = False
        else:
            # project state blocks
            self.partition.project(self.g, self.sample_actions, confidence_threshold=self.state_confidence,
                                   split_threshold=self.state_threshold, orphan_check=self.orphan_check,
                                   nth_min_state_block_confidence=self.nth_min_state_block_confidence,
                                   state_block_confidence_product=self.state_block_confidence_product,
                                   confidence_percentile=self.confidence_percentile,
                                   exclude_blocks=self.exclude_blocks, fully_convolutional=self.fully_convolutional,
                                   block_confs=self.block_confs)

            # maybe visualize state partition
            if self.visualize_state_partition is not None:
                keys, values = self.partition.get_state_partition(return_keys=True)
                print("keys for blocks:", keys)
                self.visualize_state_partition(values)

            # maybe save state partition
            if self.save_all:
                path = "{}_SB_{:d}.json".format(self.save_path, self.state_save_count + 1)
                self.save_state_partition(self.partition, path)
                self.state_save_count += 1

            some_change = False

            # iterate over all state blocks
            for state_block_index in range(len(self.partition.state_keys)):

                # split blocks with respect to the state block until convergence
                flag = True
                while flag:

                    # early termination
                    if self.max_blocks is not None and len(self.partition.blocks) >= self.max_blocks:
                        break

                    # check for split
                    flag = False
                    for new_block_index in self.partition.indices:

                        change = self.split(new_block_index, state_block_index)

                        if change:
                            flag = True
                            some_change = True
                            break

            # maybe compute block confidences
            if self.confidence_propagation:
                self.block_confs = self.get_average_confidences()

        # train state-action classifier
        if some_change:

            # average confidences
            if self.show_average_confidences:
                self.print_average_confidences()

            # maybe visualize state-action partition
            if self.visualize_state_action_partition:
                self.visualize_state_action_partition(self.partition)

            # maybe save state-action partition
            if self.save_all:
                path = "{}_B_{:d}.json".format(self.save_path, self.state_action_save_count + 1)
                self.save_state_action_partition(self.partition, path)
                self.state_action_save_count += 1

            # train the state-action block classifier
            balanced_accuracy, _, _, _ = self.train_classifier()

            # maybe set state-action num. samples threshold adaptively
            if self.adaptive_B_threshold:
                total_num_samples = np.sum([len(block) for block in self.partition.blocks])
                self.state_action_threshold = int(total_num_samples * balanced_accuracy) * \
                    self.adaptive_B_threshold_multiplier

        return some_change

    def train_classifier(self):
        """
        Train a state-action classifier.
        :return:                            Best accuracy for each class.
        """

        states = []
        actions = []
        labels = []

        if self.verbose:
            print("before de-duplication:")

        for idx, block in self.partition:

            if self.verbose:
                print("block {:d}, {:d} transitions".format(idx + 1, len(block)))

            if self.deduplicate_before_training:
                block = utils.deduplicate(block)

            for transition in block:
                states.append(transition.state)
                actions.append(transition.action)
                labels.append(idx)

        if self.reuse_blocks:
            mask = np.zeros(self.max_blocks, dtype=np.bool)
            unique_labels = set(labels)
            for unique_label in unique_labels:
                mask[unique_label] = True
        else:
            mask = None

        states = np.array(states)
        actions = np.array(actions)
        labels = np.array(labels, dtype=np.int32)

        if self.verbose:
            print()
            print("after de-duplication:")
            for idx in range(int(np.max(labels)) + 1):
                print("block {:d}: {:d} transitions".format(idx + 1, int(np.sum(labels == idx))))
            print()

        if self.reset_logits and self.g.session is not None:
            self.g.reset_last_layer()

        return self.g.fit_split(states, actions, labels, mask=mask)

    def split_rewards(self):
        """
        Split a single state-action block two or more based on rewards awarded for each task.
        :return:        True if any split occurred, otherwise False.
        """

        assert len(self.partition.blocks) == 1

        blocks = collections.defaultdict(list)

        for transition in self.partition.blocks[0]:

            if transition.reward > 0:
                key = transition.task
            else:
                key = -1

            blocks[key].append(transition)

        if len(blocks.keys()) > 1:
            self.partition.remove(0)

            for block in blocks.values():
                self.partition.add(block)

            return True
        else:
            return False

    def split(self, state_action_block_index, state_block_index):
        """
        Split a state-action block based on next state blocks.
        :param state_action_block_index:    State action block index.
        :param state_block_index:           State block index.
        :return:                            True if split occurred, otherwise False.
        """

        # split block
        block = self.partition.get(state_action_block_index)
        new_blocks = collections.defaultdict(list)
        new_blocks_counts = collections.defaultdict(lambda: 0)

        for transition in block:

            next_block_index = transition.next_state_block

            if transition.reward > 0:
                # we don't care about the next state block of the goal states
                next_block_index = 0

            key = next_block_index == state_block_index

            # ignore transitions with too low confidence in next state block prediction
            new_blocks_counts[key] += 0
            if transition.next_state_block_confidence is None or \
                    transition.next_state_block_confidence >= self.state_action_confidence:
                new_blocks[key].append(transition)
                new_blocks_counts[key] += 1
            elif not self.ignore_low_conf:
                new_blocks[key].append(transition)

        if len(new_blocks.keys()) > 1 and \
                np.all([value >= self.state_action_threshold for value in new_blocks_counts.values()]):
            # split the block
            self.partition.remove(state_action_block_index)
            for new_block in new_blocks.values():
                self.partition.add(new_block)
            return True

        # do nothing
        return False

    def get_average_confidences(self):

        block_confs = {}

        for idx, block in self.partition:

            if block[0].next_state_block_confidence is None:
                # reward block doesn't have confidence
                continue

            confs = []

            for t in block:
                confs.append(t.next_state_block_confidence)

            conf = np.mean(confs)
            block_confs[idx] = conf

        return block_confs

    def print_average_confidences(self):

        for idx, block in self.partition:

            if block[0].next_state_block_confidence is None:
                # reward block doesn't have confidence
                continue

            confs = []

            for t in block:
                confs.append(t.next_state_block_confidence)

            conf = np.mean(confs)

            if self.verbose:
                print("block {:d} ({:d}): {:.2f}% confidence".format(idx, len(block), conf * 100))
