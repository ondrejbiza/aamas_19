import collections
import time
import numpy as np
import operator
from .. import constants


class Partition:

    def __init__(self):
        """
        Partition object.
        """

        self.blocks = []

    def add(self, block):
        """
        Add a block to the partition.
        :param block:
        :return:
        """

        self.blocks.append(block)

    def get(self, index):
        """
        Get a block from the partition based on an index of the block.
        :param index:       Index of the block to get.
        :return:            None.
        """

        return self.blocks[index]

    def remove(self, index):
        """
        Remove a block from the partition based on an index of the block.
        :param index:       Index of the block to remove.
        :return:            None.
        """

        del self.blocks[index]

    def __iter__(self):

        return iter(self.blocks)

    def __len__(self):

        return len(self.blocks)


class StateActionPartition(Partition):

    MODE_PROJECT_STATE = "project_state"
    MODE_PROJECT_NEXT_STATE = "project_next_state"

    def __init__(self, classifier, sample_actions=None, fully_convolutional=False, confidence_threshold=0.0,
                 confidence_percentile=None, exclude_blocks=False, verbose=False):

        super(StateActionPartition, self).__init__()

        self.classifier = classifier
        self.sample_actions = sample_actions
        self.fully_convolutional = fully_convolutional
        self.confidence_threshold = confidence_threshold
        self.confidence_percentile = confidence_percentile
        self.exclude_blocks = exclude_blocks
        self.verbose = verbose

        self.state_keys = []
        self.state_block_counts = []
        self.distributions = None

    def project(self, single_class=False, project_all=False, block_confidences=None):

        self.state_keys = [constants.GOAL_STATE]
        self.state_block_counts = [0]

        for mode in [self.MODE_PROJECT_NEXT_STATE, self.MODE_PROJECT_STATE]:

            if not project_all and mode == self.MODE_PROJECT_STATE:
                continue

            for block_idx, block in enumerate(self.blocks):

                states = []

                for transition in block:
                    if mode == self.MODE_PROJECT_NEXT_STATE:
                        states.append(transition.next_state)
                    else:
                        states.append(transition.state)

                start = time.time()
                predictions = self.batch_predict(states)
                duration = time.time() - start

                if self.verbose:
                    print()
                    print("block {:d} predict: {:.2f} seconds".format(block_idx, duration))

                start = time.time()
                for transition, prediction in zip(block, predictions):

                    if mode == self.MODE_PROJECT_NEXT_STATE and transition.is_end and transition.reward > 0:
                        # transitions into the goal state, ignore
                        transition.next_state_block = self.state_keys.index(constants.GOAL_STATE)
                        transition.next_state_block_confidence = 1.0
                    else:
                        # classify next state
                        if single_class:
                            blocks = {0}
                            confidence = 1.0
                        else:
                            blocks, confidence = self.evaluate_prediction(
                                prediction, block_confidences=block_confidences
                            )

                        key = frozenset(blocks)
                        if key not in self.state_keys:
                            next_state_block_index = len(self.state_keys)
                            self.state_keys.append(key)
                            self.state_block_counts.append(1)
                        else:
                            next_state_block_index = self.state_keys.index(key)
                            self.state_block_counts[next_state_block_index] += 1

                        if mode == self.MODE_PROJECT_NEXT_STATE:
                            transition.next_state_block = next_state_block_index
                            transition.next_state_block_confidence = confidence
                        else:
                            transition.state_block = next_state_block_index
                            transition.state_block_confidence = confidence

                duration = time.time() - start

                if self.verbose:
                    print()
                    print("block {:d} evaluate: {:.2f} seconds".format(block_idx, duration))

    def batch_predict(self, states):

        if self.fully_convolutional:
            predictions = self.classifier.predict_all_actions(states)
        else:
            # sample all actions and evaluate them using the standard convnet
            predictions = []

            for state in states:

                sampled_actions = self.sample_actions(state)

                tmp_predictions = self.classifier.predict(
                    [state for _ in range(len(sampled_actions))], sampled_actions
                )

                predictions.append(tmp_predictions)

            predictions = np.stack(predictions, axis=0)

        return predictions

    def evaluate_prediction(self, action_predictions, block_confidences=None):

        assert len(action_predictions.shape) == 2

        blocks = np.argmax(action_predictions, axis=1)
        probabilities = action_predictions[list(range(len(blocks))), blocks]

        if block_confidences is not None:
            probabilities *= block_confidences[blocks]

        if self.exclude_blocks:
            blocks = blocks[probabilities >= self.confidence_threshold]

        blocks = set(blocks)

        if self.confidence_percentile is not None:
            confidence = np.percentile(probabilities, self.confidence_percentile)
        else:
            confidence = np.min(probabilities)

        return blocks, confidence

    def get_full_state_partition(self, return_keys=False):
        """
        Get next states from transitions sorted according to their state blocks.
        :param return_keys:         Return block keys.
        :return:                    State partition as an array.
        """

        state_partition = collections.defaultdict(list)

        for block in self.blocks:
            for transition in block:
                state_partition[transition.next_state_block].append(transition.next_state)

        keys = []
        values = []

        for key, value in state_partition.items():
            keys.append(self.state_keys[key])
            values.append(value)

        if return_keys:
            return keys, values
        else:
            return values

    def get_state_block_confidences(self):

        state_blocks_dict = collections.defaultdict(list)

        for block in self.blocks:
            for transition in block:
                state_blocks_dict[transition.next_state_block].append(transition.next_state_block_confidence)

        for key in state_blocks_dict.keys():
            state_blocks_dict[key] = np.mean(state_blocks_dict[key])

        return state_blocks_dict

    def sort_state_block_indices_by_most_confident(self):

        state_block_confidences = self.get_state_block_confidences()
        sorted_confidences = sorted(state_block_confidences.items(), key=operator.itemgetter(1), reverse=True)
        state_block_indices = [x[0] for x in sorted_confidences]

        return state_block_indices

    def sort_state_action_blocks_by_max_representative(self, state_block_idx):

        counts = {idx: 0 for idx in range(len(self.blocks))}

        for block_idx, block in enumerate(self.blocks):
            for transition in block:
                if transition.next_state_block == state_block_idx:
                    counts[block_idx] += 1

        sorted_confidences = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        state_block_indices = [x[0] for x in sorted_confidences]

        return state_block_indices

    def get_majority_state_partition(self, return_keys=False):
        """
        Get indices of state blocks that have a state-block block leading to them.
        :param return_keys:         Return state block keys.
        :return:                    State block indices and maybe keys.
        """

        successor_table = self.get_state_action_successor_table()

        state_blocks = list(successor_table.values())

        if return_keys:
            state_keys = []

            for block in state_blocks:
                state_keys.append(self.state_keys[block])

            return state_keys, state_blocks
        else:
            return state_blocks

    def get_state_action_successor_table(self, return_match=False):
        """
        Get a table of next state blocks for each state-action block.
        :param return_match:        Return fraction of votes that are in majority.
        :return:        Table.
        """

        t = {}
        match = []

        for idx, block in enumerate(self.blocks):

            # perform majority voting
            votes = collections.defaultdict(lambda: 0)

            for transition in block:
                votes[transition.next_state_block] += 1

            winner_tuple = max(votes.items(), key=operator.itemgetter(1))
            t[idx] = winner_tuple[0]

            if return_match:
                # get fraction of votes in majority
                hits = winner_tuple[1]
                total = len(block)
                match.append(hits / total)

        if return_match:
            # also return fraction of votes in the majority averaged over all state-action blocks
            return t, match
        else:
            return t

    def get_state_successor_table(self):
        """
        Get a table of a list of next available state-action blocks for each state block.
        :return:        Table.
        """

        t = {}

        for i in range(len(self.state_keys)):

            t[i] = self.state_keys[i]

        return t

    def get_goal_state_action_block(self, task_idx=None):
        """
        Get goal state-action block (assuming there is exactly one such partition).
        :return:        Goal state-action block index.
        """

        for idx, block in enumerate(self.blocks):

            if block[0].reward > 0 and (task_idx is None or block[0].task == task_idx):
                return idx

        return None

    def get_state_action_overlap(self, model):
        """
        Get overlap between predicted state-action partition and the ground-truth partition.
        :param model:       Ground-truth abstract model.
        :return:            Overlap and total number of experience.
        """

        pred_count = len(self.blocks)

        # compute matches between predicted and ground-truth state-action partitions
        num_matches = np.zeros((len(model.state_action_blocks), pred_count), dtype=np.int32)

        for gt_idx, block_id in enumerate(model.state_action_blocks):

            for pred_idx in range(pred_count):

                for transition in self.blocks[pred_idx]:

                    gt_block = model.get_state_action_block(transition.state, transition.next_state)

                    if gt_block == block_id:

                        num_matches[gt_idx][pred_idx] += 1

        total = np.sum(num_matches)

        # calculate overlap
        matches = {}
        hits = 0

        for _ in range(min(len(model.state_action_blocks), pred_count)):

            coords = np.unravel_index(num_matches.argmax(), num_matches.shape)
            matches[coords[0]] = coords[1]

            hits += num_matches[coords]

            num_matches[coords[0], :] = -1
            num_matches[:, coords[1]] = -1

        return hits, total

    def get_state_overlap(self, model):
        """
        Get overlap between predicted state partition and the ground-truth partition.
        :param model:       Ground-truth abstract model.
        :return:            Overlap and total number of experience.
        """

        gt_count = model.NUM_STATE_BLOCKS
        pred_count = len(self.state_keys)

        # compute matches between predicted and ground-truth state-action partitions
        num_matches = np.zeros((gt_count, pred_count), dtype=np.int32)

        for gt_idx in range(gt_count):

            for block in self.blocks:

                for transition in block:

                    gt_block = model.get_state_block(transition.next_state)

                    if gt_block == gt_idx:
                        num_matches[gt_idx][transition.next_state_block] += 1

        total = np.sum(num_matches)

        # calculate overlap
        matches = {}
        hits = 0

        for _ in range(min(gt_count, pred_count)):

            coords = np.unravel_index(num_matches.argmax(), num_matches.shape)
            matches[coords[0]] = coords[1]

            hits += num_matches[coords]

            num_matches[coords[0], :] = -1
            num_matches[:, coords[1]] = -1

        return hits, total


class StateActionPartitionFactory:

    @staticmethod
    def from_list(partition_list):
        """
        Construct a partition from an array of state-action blocks.
        :param partition_list:      List of state-action blocks.
        :return:                    Partition object.
        """

        partition = StateActionPartition(None)

        for block in partition_list:
            partition.add(block)

        return partition
