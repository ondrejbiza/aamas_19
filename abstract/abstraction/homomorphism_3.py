import collections
import numpy as np
from .partition_3 import StateActionPartition
from . import utils


class Homomorphism:

    def __init__(self, experience, classifier, max_blocks, sample_actions=None, fully_convolutional=False,
                 size_threshold=0, confidence_threshold=0.0, confidence_percentile=None, confidence_propagation=False,
                 ignore_low_conf=False, exclude_blocks=False, deduplicate_before_training=False,
                 show_average_confidences=False, reuse_network=True, verbose=False):

        self.experience = experience
        self.classifier = classifier

        self.sample_actions = sample_actions
        self.fully_convolutional = fully_convolutional

        self.size_threshold = size_threshold
        self.confidence_threshold = confidence_threshold
        self.confidence_percentile = confidence_percentile
        self.confidence_propagation = confidence_propagation
        self.ignore_low_conf = ignore_low_conf
        self.exclude_blocks = exclude_blocks

        self.max_blocks = max_blocks
        self.verbose = verbose
        self.show_average_confidences = show_average_confidences
        self.deduplicate_before_training = deduplicate_before_training
        self.reuse_network = reuse_network

        self.first_step = True
        self.block_confidences = None

        self.partition = StateActionPartition(
            classifier, sample_actions=sample_actions, fully_convolutional=fully_convolutional,
            confidence_threshold=confidence_threshold, confidence_percentile=confidence_percentile,
            exclude_blocks=exclude_blocks, verbose=verbose
        )
        self.partition.add(experience)

    def partition_iteration(self):
        """
        Run partition iteration.
        :return:        Number of steps before convergence (or before the limit was reached).
        """

        step = 0
        change = True

        while change:
            change = self.partition_improvement()

            if self.verbose:
                print("block sizes:")
                for idx, block in enumerate(self.partition.blocks):
                    print("{:d}: {:d} samples".format(idx, len(block)))

            step += 1

            # early termination
            if self.max_blocks is not None and len(self.partition.blocks) >= self.max_blocks:
                break

        # project all at the end
        self.partition.project(
            block_confidences=self.block_confidences, project_all=True
        )

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
            self.partition.project(
                block_confidences=self.block_confidences
            )

            some_change = False

            # iterate over all state blocks
            for state_block_index in range(len(self.partition.state_keys)):

                # split blocks with respect to the state block until convergence
                flag = True
                step = 0
                while flag:

                    # early termination
                    if self.max_blocks is not None and len(self.partition.blocks) >= self.max_blocks:
                        break

                    # check for split
                    flag = False

                    for new_block_index in range(len(self.partition)):

                        change = self.split(new_block_index, state_block_index)

                        if change:
                            flag = True
                            some_change = True
                            step += 1
                            break

        # train state-action classifier
        if some_change:

            # average confidences
            if self.show_average_confidences:
                self.print_average_confidences()

            # train the state-action block classifier
            self.train_classifier()

        return some_change

    def train_classifier(self):
        """
        Train a state-action classifier.
        :return:                            None.
        """

        # get training data
        states = []
        actions = []
        labels = []

        for idx, block in enumerate(self.partition):

            if self.deduplicate_before_training:
                block = utils.deduplicate(block)

            for transition in block:
                states.append(transition.state)
                actions.append(transition.action)
                labels.append(idx)

        if self.reuse_network:

            if self.classifier.session is not None and not self.fully_convolutional:
                # reset logits, not yet implemented in fully-conv network
                self.classifier.reset_last_layer()

            # create a new mask
            mask = np.zeros(self.max_blocks, dtype=np.bool)
            unique_labels = set(labels)
            for unique_label in unique_labels:
                mask[unique_label] = True
        else:
            mask = None

        states = np.array(states)
        actions = np.array(actions)
        labels = np.array(labels, dtype=np.int32)

        # train the classifier
        self.classifier.fit_split(states, actions, labels, mask=mask)

    def split_rewards(self):
        """
        Split a single state-action block two or more based on the received rewards.
        :return:        True if any split occurred, otherwise False.
        """

        assert len(self.partition.blocks) == 1

        blocks = collections.defaultdict(list)

        for transition in self.partition.blocks[0]:

            blocks[transition.reward].append(transition)

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
                    transition.next_state_block_confidence >= self.confidence_threshold:
                new_blocks[key].append(transition)
                new_blocks_counts[key] += 1
            elif not self.ignore_low_conf:
                new_blocks[key].append(transition)

        if self.verbose:
            print("Trying to split state-action block {:d} w.r.t. state block {}."
                  .format(state_action_block_index, self.partition.state_keys[state_block_index]))
            print("Block counts:", new_blocks_counts)

        if len(new_blocks.keys()) > 1 and \
                np.all([value >= self.size_threshold for value in new_blocks_counts.values()]):
            # split the block
            self.partition.remove(state_action_block_index)
            for new_block in new_blocks.values():
                self.partition.add(new_block)
            return True

        # do nothing
        return False

    def print_average_confidences(self):

        for idx, block in enumerate(self.partition):

            if block[0].next_state_block_confidence is None:
                # reward block doesn't have confidence
                continue

            confs = []

            for t in block:
                confs.append(t.next_state_block_confidence)

            conf = np.mean(confs)

            print("block {:d} ({:d}): {:.2f}% confidence".format(idx, len(block), conf * 100))
