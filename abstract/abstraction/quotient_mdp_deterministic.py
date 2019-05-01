import copy as cp
import collections
from io import BytesIO
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .. import constants
from .. import vis_utils


class QuotientMDPDeterministic:

    # quotient MDP plot
    NODE_FILL_COLOR = "#E1D5E7"
    NODE_BORDER_COLOR = "#9674A6"
    STANDARD_EDGE_COLOR = "#666666"
    REWARD_EDGE_COLOR = "#8CB972"
    RESOLUTION = 500        # dpi

    # partition report
    BLOCK_NUM_ROWS = 10
    BLOCK_NUM_COLS = 5

    # value iteration
    DIFF_THRESHOLD = 0.001
    MAX_STEPS = 1000

    def __init__(self, states, actions, allowed_actions, transitions, rewards, discount=0.99):

        self.states = states
        self.actions = actions
        self.allowed_actions = allowed_actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount = discount

        self.state_action_values = np.zeros((len(self.states), len(self.actions)), dtype=np.float32)
        self.policy = np.zeros(len(self.states), dtype=np.int32)

    def value_iteration(self):

        max_diff = self.DIFF_THRESHOLD + 1
        step = 0

        while max_diff > self.DIFF_THRESHOLD and step < self.MAX_STEPS:

            new_state_action_values = np.zeros_like(self.state_action_values)

            for state_idx, state in enumerate(self.states):

                for action_idx, action in enumerate(self.actions):

                    if action not in self.allowed_actions[state]:
                        continue

                    next_state = self.transitions[(state, action)]
                    next_state_idx = self.states.index(next_state)

                    new_state_action_values[state_idx, action_idx] = \
                        self.rewards[(state, action)] + \
                        self.discount * np.max(self.state_action_values[next_state_idx, :])

            max_diff = np.max(np.abs(self.state_action_values - new_state_action_values))

            self.state_action_values = new_state_action_values

            step += 1

        return step

    def get_state_action_block_q_values(self):

        q_values = {action: [] for action in self.actions}

        # collect q-values of state-action blocks for all state blocks
        for state_idx, state in enumerate(self.states):

            for action_idx, action in enumerate(self.actions):

                if action not in self.allowed_actions[state]:
                    continue

                q_values[action].append(self.state_action_values[state_idx, action_idx])

        # all collected q-values for a single state-action block should be the same, but we average them just in case
        for action in self.actions:

            if len(q_values[action]) > 0:
                q_values[action] = np.mean(q_values[action])
            else:
                # this action cannot be executed from any state => 0 value
                q_values[action] = 0

        return q_values

    def partition_report(self, partition, classifier, save_path, num_actions=None, stride=None):

        mdp_fig = self.plot_mdp(save_path=None, return_figure=True)
        state_action_figs = self.plot_state_action_blocks(partition, num_actions=num_actions, stride=stride)
        state_figs = self.plot_state_blocks(partition, classifier)

        # save plots
        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(save_path)

        pdf.savefig(mdp_fig, dpi=self.RESOLUTION)

        for fig in state_action_figs:
            pdf.savefig(fig)

        for fig in state_figs:
            pdf.savefig(fig)

        pdf.close()

    def plot_mdp(self, save_path=None, return_figure=False):

        import pydot
        graph = pydot.Dot(graph_type="digraph")

        for state in self.states:

            state_name = self.get_state_name(state)

            node = pydot.Node(state_name, fillcolor=self.NODE_FILL_COLOR, color=self.NODE_BORDER_COLOR,
                              style="filled")
            graph.add_node(node)

            for action in self.allowed_actions[state]:

                next_state = self.transitions[(state, action)]
                next_state_name = self.get_state_name(next_state)
                reward = self.rewards[(state, action)]

                if reward > 0:
                    color = self.REWARD_EDGE_COLOR
                else:
                    color = self.STANDARD_EDGE_COLOR

                edge = pydot.Edge(state_name, next_state_name, label=str(action), color=color)
                graph.add_edge(edge)

        if save_path is not None:
            assert save_path.split(".")[-1] == "pdf"
            graph.write_pdf(save_path)

        if return_figure:
            # return figure so that we can save more figures into the same pdf file
            png = graph.create_png(prog=["dot", "-Gdpi={:d}".format(self.RESOLUTION)])

            bio = BytesIO()
            bio.write(png)
            bio.seek(0)
            img = mpimg.imread(bio)

            fig, ax = plt.subplots()
            ax.imshow(img, aspect="equal")
            ax.axis("off")

            return fig

    def plot_state_blocks(self, partition, classifier):

        figures = []

        for state_idx, state_key in enumerate(partition.state_keys):

            states = []

            for block in partition.blocks:

                for transition in block:

                    if transition.state_block == state_idx:
                        states.append(transition.state)

                    if transition.is_end and transition.next_state_block == state_idx:
                        states.append(transition.next_state)

            # deduplicate with counts
            counts = collections.defaultdict(lambda: [None, 0])
            for state in states:
                key = tuple(state[0].reshape(-1, ).tolist()) + (state[1],)
                counts[key] = [state, counts[key][1] + 1]

            to_show = list(counts.values())

            # plot
            num_samples = len(to_show)
            total_num_samples = int(np.sum([payload[1] for payload in to_show]))

            figure_name = "state block {} ({:d} samples, {:d} unique)".format(
                self.get_state_name(state_key), total_num_samples, num_samples
            )

            if num_samples == 0:
                continue

            num_rows = int(np.ceil(float(num_samples) / self.BLOCK_NUM_COLS))
            num_rows = min(num_rows, self.BLOCK_NUM_ROWS)

            fig, axes = plt.subplots(nrows=num_rows * 2, ncols=self.BLOCK_NUM_COLS)

            if len(axes.shape) == 1:
                axes = np.expand_dims(axes, axis=0)

            fig.set_figheight(24)
            fig.set_figwidth(6)
            fig.suptitle(figure_name)

            for t_idx, payload in enumerate(to_show):

                if t_idx >= num_rows * self.BLOCK_NUM_COLS:
                    break

                state, count = payload
                state = cp.deepcopy(state)

                predictions = classifier.predict_all_actions([state], apply_softmax=True, flat=True)[0]

                classes = np.argmax(predictions, axis=1)

                values = classes + predictions[list(range(len(classes))), classes]
                values = np.reshape(values, (state[0].shape[0], state[0].shape[1]))

                image = vis_utils.multiplex_hand_state(state, 3)

                axis = axes[(t_idx // self.BLOCK_NUM_COLS) * 2, t_idx % self.BLOCK_NUM_COLS]
                axis.imshow(image[:, :, 0], vmin=0, vmax=3)
                axis.annotate(count, (5, 16), color="white", fontsize=10)
                axis.axis("off")

                axis = axes[(t_idx // self.BLOCK_NUM_COLS) * 2 + 1, t_idx % self.BLOCK_NUM_COLS]
                im = axis.imshow(values, vmin=0, vmax=np.max(classes) + 1)

                divider = make_axes_locatable(axis)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                cbar.ax.tick_params(labelsize=5)

                axis.axis("off")

            # treat empty boxes
            for idx in range(max(0, num_rows * self.BLOCK_NUM_COLS - len(to_show))):
                idx = num_rows * self.BLOCK_NUM_COLS - idx - 1

                axis = axes[idx // self.BLOCK_NUM_COLS, idx % self.BLOCK_NUM_COLS]
                axis.imshow(np.zeros_like(to_show[0][0][0][:, :, 0]), vmin=0, vmax=3)
                axis.axis("off")

            figures.append(fig)

        # return figures so that we can save them together with the MDP schema into a single pdf
        return figures

    def plot_state_action_blocks(self, partition, num_actions=None, stride=None):

        figures = []

        for block_idx, block in enumerate(partition.blocks):

            # deduplicate with counts
            counts = collections.defaultdict(lambda: [None, 0])
            for transition in block:
                key = tuple(transition.state[0].reshape(-1, ).tolist()) + (transition.state[1], transition.action)
                counts[key] = [transition, counts[key][1] + 1]

            to_show = list(counts.values())

            # plot
            num_samples = len(to_show)
            total_num_samples = int(np.sum([payload[1] for payload in to_show]))

            figure_name = "state-action block {:d} ({:d} samples, {:d} unique)".format(
                block_idx, total_num_samples, num_samples
            )

            num_rows = int(np.ceil(num_samples / self.BLOCK_NUM_COLS))
            num_rows = min(num_rows, self.BLOCK_NUM_ROWS) * 2

            if num_samples == 0:
                continue

            fig, axes = plt.subplots(nrows=num_rows, ncols=self.BLOCK_NUM_COLS)

            if len(axes.shape) == 1:
                axes = np.expand_dims(axes, axis=0)

            fig.set_figheight(24)
            fig.set_figwidth(6)
            fig.suptitle(figure_name)

            for t_idx, payload in enumerate(to_show):

                if t_idx >= num_rows * self.BLOCK_NUM_COLS:
                    break

                t, count = payload
                t = cp.deepcopy(t)

                state = t.state
                action = t.action

                image = vis_utils.multiplex_hand_state(state, 3)

                if num_actions is not None and stride is not None:
                    image = vis_utils.multiplex_action(
                        num_actions ** 2, num_actions, 28, stride, image, action, 3,
                        padding=112 / (num_actions * 2), column_first=False
                    )

                axis = axes[t_idx // self.BLOCK_NUM_COLS, t_idx % self.BLOCK_NUM_COLS]
                axis.imshow(image[:, :, 0], vmin=0, vmax=3)
                axis.annotate(count, (5, 16), color="white", fontsize=10)
                axis.axis("off")

            # treat empty boxes
            for idx in range(max(0, num_rows * self.BLOCK_NUM_COLS - len(to_show))):
                idx = num_rows * self.BLOCK_NUM_COLS - idx - 1

                axis = axes[idx // self.BLOCK_NUM_COLS, idx % self.BLOCK_NUM_COLS]
                axis.imshow(np.zeros_like(to_show[0][0].state[0][:, :, 0]), vmin=0, vmax=3)
                axis.axis("off")

            figures.append(fig)

        # return figures so that we can save them together with the MDP schema into a single pdf
        return figures

    @staticmethod
    def get_state_name(state):

        if isinstance(state, str):
            return state
        else:
            return str(set(state))


class QuotientMDPFactory:

    @staticmethod
    def from_partition(partition):

        table = partition.get_state_action_successor_table()
        goal = partition.get_goal_state_action_block()

        states = list(partition.state_keys)
        actions = list(sorted(list(table.keys())))

        allowed_actions = {}
        transitions = {}
        rewards = {}

        for state in states:

            allowed_actions[state] = []

            if state == constants.GOAL_STATE:
                continue

            for block in state:

                assert block in actions

                allowed_actions[state].append(block)
                transitions[(state, block)] = partition.state_keys[table[block]]

                if goal == block:
                    rewards[(state, block)] = 10.0
                else:
                    rewards[(state, block)] = 0.0

        return QuotientMDPDeterministic(states, actions, allowed_actions, transitions, rewards)
