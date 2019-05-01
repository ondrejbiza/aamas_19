import bisect
import collections
import json
import numpy as np
import operator
from .. import algorithms
from . import utils


class Partition:

    def __init__(self, max_blocks=10000):
        """
        Partition object.
        """

        self.free_indices = list(range(max_blocks))
        self.indices = []
        self.blocks = []

    def add(self, block):
        """
        Add a block to the partition.
        :param block:
        :return:
        """

        self.blocks.append(block)
        self.indices.append(self.free_indices[0])
        del self.free_indices[0]

    def get(self, index):
        """
        Get a block from the partition based on an index of the block.
        :param index:       Index of the block to get.
        :return:            None.
        """

        block_idx = self.indices.index(index)
        return self.blocks[block_idx]

    def remove(self, index):
        """
        Remove a block from the partition based on an index of the block.
        :param index:       Index of the block to remove.
        :return:            None.
        """

        block_idx = self.indices.index(index)
        del self.blocks[block_idx]
        del self.indices[block_idx]
        bisect.insort(self.free_indices, index)

    def __eq__(self, other):

        return self.indices == other.indices

    def __ne__(self, other):

        return not self.__eq__(other)

    def __iter__(self):

        return zip(self.indices, self.blocks)

    def __len__(self):

        return len(self.indices)


class StateActionPartition(Partition):

    def __init__(self, max_blocks=10000):

        super(StateActionPartition, self).__init__(max_blocks=max_blocks)
        self.state_keys = []
        self.state_block_counts = []
        self.distributions = None

    def project(self, classifier, sample_actions, single_class=False, confidence_threshold=0.0, split_threshold=0,
                orphan_check=False, nth_min_state_block_confidence=False, state_block_confidence_product=False,
                confidence_percentile=None, exclude_blocks=False, fully_convolutional=False, block_confs=None):
        """
        Create a state partition using a state-action block classifier.
        Instead of creating an explicit state partition, the index of the state block each transition goes to
        is added as the fifth item in the transition tuple: (state, action, reward, next_state, done,
        next_state_block_index).
        :param classifier:                      State-action block classifier.
        :param sample_actions:                  Function that samples actions.
        :param single_class:                    There is only one state-action block.
        :param confidence_threshold             Ignore predictions with confidence below this value.
        :param split_threshold:                 Minimum number of samples in a state block.
        :param orphan_check:                    Check for orphaned state blocks.
        :param nth_min_state_block_confidence:  Use nth minimum when calculating block confidence.
        :param state_block_confidence_product:  Use the product of confidences when calculating block confidence.
        :return:                                None.
        """

        self.state_keys = []
        self.state_block_counts = []

        for block in self.blocks:

            for transition in block:

                if transition.is_end and transition.reward > 0:
                    # transitions into the goal state, ignore
                    transition.next_state_block = None
                    transition.next_state_block_confidence = 1.0
                else:
                    # classify next state
                    if single_class:
                        blocks = {0}
                        confidence = 1
                    else:

                        if fully_convolutional:
                            blocks, confidence = utils.get_next_state_action_blocks_3(
                                transition.next_state, classifier,
                                confidence_threshold=confidence_threshold,
                                confidence_percentile=confidence_percentile,
                                exclude_blocks=exclude_blocks,
                                block_confs=block_confs
                            )
                        else:
                            blocks, confidence = utils.get_next_state_action_blocks_2(
                                transition.next_state, classifier, sample_actions,
                                confidence_threshold=confidence_threshold,
                                confidence_percentile=confidence_percentile,
                                exclude_blocks=exclude_blocks,
                                block_confs=block_confs
                            )

                    key = frozenset(blocks)
                    if key not in self.state_keys:
                        next_state_block_index = len(self.state_keys)
                        self.state_keys.append(key)
                        self.state_block_counts.append(1)
                    else:
                        next_state_block_index = self.state_keys.index(key)
                        self.state_block_counts[next_state_block_index] += 1

                    transition.next_state_block = next_state_block_index
                    transition.next_state_block_confidence = confidence

        # merge blocks that are below threshold
        self.state_block_counts = np.array(self.state_block_counts, dtype=np.int32)
        below_threshold = self.state_block_counts < split_threshold

        if orphan_check:
            t = self.get_state_action_successor_table()
            for idx in range(len(below_threshold)):
                if idx not in t.values():
                    below_threshold[idx] = True

        if np.all(below_threshold):
            # merge into a single block
            self.state_keys = [frozenset(self.indices)]
            self.state_block_counts = [0]
            for block in self.blocks:
                for transition in block:
                    transition.next_state_block = 0
                    self.state_block_counts[0] += 1

        elif np.any(below_threshold):

            bad_state_keys = [(idx, item) for idx, item in enumerate(self.state_keys) if below_threshold[idx]]

            new_state_keys = []
            replacement_dict = {}

            for idx, item in enumerate(self.state_keys):

                if not below_threshold[idx]:

                    replacement_dict[idx] = len(new_state_keys)
                    new_state_keys.append(item)

            self.state_keys = new_state_keys
            self.state_block_counts = [count for idx, count in enumerate(self.state_block_counts)
                                       if not below_threshold[idx]]

            for i, key in bad_state_keys:

                min_key = None
                min_distance = None

                for j in range(len(self.state_keys)):

                    distance = algorithms.edit_distance(list(sorted(key)), list(sorted(self.state_keys[j])))

                    if min_distance is None or min_distance > distance:
                        min_distance = distance
                        min_key = j

                replacement_dict[i] = min_key

            for block in self.blocks:
                for transition in block:
                    if transition.next_state_block in replacement_dict.keys():
                        transition.next_state_block = replacement_dict[transition.next_state_block]
                        self.state_block_counts[transition.next_state_block] += 1

    def get_state_partition(self, return_keys=False):
        """
        Get state partition.
        :param return_keys:         Return block keys.
        :return:                    State partition as an array.
        """

        state_partition = {}

        for block in self.blocks:
            for transition in block:

                if transition.next_state_block not in state_partition:
                    state_partition[transition.next_state_block] = []

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

    def get_state_action_successor_table(self, majority_voting=True, return_match=False):
        """
        Get a table of next state blocks for each state-action block.
        :param majority_voting:     Vote for the next state block.
        :param return_match:        Return fraction of votes that are in majority.
        :return:        Table.
        """

        assert (not return_match) or majority_voting

        t = {}
        match = []

        for idx, block in zip(self.indices, self.blocks):

            if majority_voting:
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

            else:
                # believe the first vote
                t[idx] = block[0].next_state_block

        if return_match:
            # also return fraction of votes in the majority averaged over all state-action blocks
            return t, np.mean(match)
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

        for idx, block in zip(self.indices, self.blocks):

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

    def visualize_quotient_mdp(self, save_path=None):
        """
        Draw a graph of the quotient MDP induced by this partition.
        :param save_path:       Save path for the image.
        :return:                None.
        """

        import graph_tool.all as gt

        # add intermediate states
        g = gt.Graph(directed=True)
        vertices = [g.add_vertex() for _ in range(len(self.state_keys))]

        # add goal state
        goal_vertex = g.add_vertex()

        t = self.get_state_action_successor_table()

        for state_idx, state_key in enumerate(self.state_keys):
            for state_action_idx in state_key:

                start = vertices[state_idx]
                end_idx = t[state_action_idx]

                if end_idx is None:
                    end = goal_vertex
                else:
                    end = vertices[end_idx]

                g.add_edge(start, end)

        pos = gt.sfdp_layout(g)
        gt.graph_draw(g, pos, vertex_text=g.vertex_index, vertex_font_size=18, vertex_size=50,
                      output_size=(1000, 1000), output=save_path, bg_color=[1, 1, 1, 1])

    def export_state_action_partition_json(self, get_block, colors_list, path):
        """
        Export state-action partition into a JSON file that can be visualized with viz/show_partition.html.
        :param get_block:       Function that returns a ground-truth state-action block given the current
                                and next state.
        :param colors_list:     List of colors for each ground-truth state-action partition.
        :param path:            Save path.
        :return:                None.
        """

        d = {
            "children": []
        }

        for idx in sorted(self.indices):
            block = self.get(idx)

            d["children"].append({
                "name": "block {:d}".format(idx + 1),
                "children": []
            })

            for transition in block:

                color = colors_list[get_block(transition.state, transition.next_state)]
                d["children"][idx]["children"].append({
                    "size": 1,
                    "color": color
                })

        with open(path, "w") as file:
            json.dump(d, file)

    def export_state_partition_json(self, get_block, colors_list, path):
        """
        Export state partition into a JSON file that can be visualized with viz/show_partition.html.
        :param get_block:       Function that returns a state block given the current state.
        :param colors_list:     List of colors for each ground-truth state block.
        :param path:            Save path.
        :return:                None.
        """

        d = {
            "children": []
        }

        for idx in range(len(self.state_keys)):

            name = "block {:d}, keys: ".format(idx + 1)
            for key in self.state_keys[idx]:
                name += "{}, ".format(key + 1)
            name = name[:-2]

            d["children"].append({
                "name": name,
                "children": []
            })

        for block in self.blocks:
            for transition in block:
                if transition.next_state_block is None:
                    continue

                color = colors_list[get_block(transition.next_state)]
                d["children"][transition.next_state_block]["children"].append({
                    "size": 1,
                    "color": color
                })

        with open(path, "w") as file:
            json.dump(d, file)

    def get_multinoulli_distributions(self, prior_count=0):
        """
        Get multinoulli distribution for each state block.
        :param prior_count:     Prior strength.
        :return:        None.
        """

        self.distributions = []
        num_state_blocks = len(self.state_keys)

        for block_idx in self.indices:

            block = self.get(block_idx)

            distribution = np.zeros(num_state_blocks) + prior_count
            total = 0

            for transition in block:
                distribution[transition.next_state_block] += 1
                total += 1

            distribution /= total
            self.distributions.append(distribution)

    def get_entropies(self):
        """
        Get entropy of each state block based on their multinoulli distributions.
        :return:        List of entropies; ordered by block indices (self.indices).
        """

        if self.distributions is None or len(self.distributions) != len(self.indices):
            self.get_multinoulli_distributions()

        entropies = []

        for block_idx in range(len(self.distributions)):

            probs_above_zero = self.distributions[block_idx][self.distributions[block_idx] > 0]

            entropy = - np.sum(probs_above_zero * np.log2(probs_above_zero))
            entropies.append(entropy)

        return entropies


class PartitionFactory:

    @staticmethod
    def from_list(partition_list):
        """
        Construct a partition from an array of state-action blocks.
        :param partition_list:      List of state-action blocks.
        :return:                    Partition object.
        """

        partition = StateActionPartition()

        for block in partition_list:
            partition.add(block)

        return partition
