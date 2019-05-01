import os
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm


class Animator:

    def __init__(self, wait_time=None, max_value=3):
        """
        Animates a set of images that can be supplied iteratively.
        :param wait_time:       Block for this amount of time. Pass None for no blocking.
        :param max_value:       Maximum value in the image.
        """

        self.wait_time = wait_time
        self.max_value = max_value
        self.canvas = None

    def next(self, image):
        """
        Draw the next image.
        :param image:       An image.
        :return:
        """

        if self.canvas is None:
            self.canvas = plt.imshow(image, vmin=0, vmax=self.max_value)
            plt.colorbar()
            plt.axis("off")
        else:
            self.canvas.set_data(image)
            plt.draw()

        if self.wait_time is not None:
            plt.pause(self.wait_time)

    @staticmethod
    def close():
        """
        Close the figure.
        :return:
        """

        plt.close()


class Saver:

    FILE_TEMPLATE = "episode_{:d}.gif"

    def __init__(self, max_value=3, gif_interval=700, dir_path=None):
        """
        Initialize a saver for visualizing grid world environments.
        :param max_value:       Maximum value in the image.
        :param gif_interval:    Delay between frames in an animation.
        :param dir_path:        Directory where to save the gifs.
        """

        self.max_value = max_value
        self.gif_interval = gif_interval
        self.dir_path = dir_path

        self.frames = None
        self.episode_idx = 1
        self.reset()

    def add(self, image):
        """
        Add a depth image.
        :param image:       A depth image.
        :return:            None.
        """

        self.frames.append(cp.deepcopy(image))

    def save_frames(self, dir_path, template="frame_{:d}.svg"):
        """
        Save added images.
        :param dir_path:    Directory path.
        :param template:    Template for image names.
        :return:
        """

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        for idx, frame in enumerate(self.frames):

            save_path = os.path.join(dir_path, template.format(idx + 1))

            plt.imshow(frame, vmin=0, vmax=self.max_value)
            plt.colorbar()
            plt.axis("off")
            plt.savefig(save_path)
            plt.close()

    def save_gif(self, file_path=None):
        """
        Save a gif of all images.
        :param file_path:       Save path for the gif.
        :return:                None.
        """

        if file_path is None:

            if not os.path.isdir(self.dir_path):
                os.makedirs(self.dir_path)

            file_path = os.path.join(self.dir_path, self.FILE_TEMPLATE.format(self.episode_idx))

        fig = plt.figure()
        ax = plt.gca()

        canvas = plt.imshow(self.frames[0], vmin=0, vmax=self.max_value)
        text = ax.annotate(0, (5, 13), color="white", fontsize=13)
        plt.colorbar()
        plt.axis("off")
        plt.tight_layout()

        def update(i):
            canvas.set_data(self.frames[i])
            text.set_text(str(i + 1))
            return canvas, ax

        anim = FuncAnimation(fig, update, frames=np.arange(0, len(self.frames)), interval=self.gif_interval)
        anim.save(file_path, dpi=80, writer="imagemagick")
        plt.close()

    def reset(self):
        """
        Reset the saver.
        :return:                None.
        """

        self.frames = []
        self.episode_idx += 1


def visualize_state_partition(state_partition, num_cells=4, vmin=0, vmax=2, process_state=None, show=True):
    """
    Visualize state partition.
    :param state_partition:     State partition.
    :param num_cells:           Number of cells to show.
    :param vmin:                Minimum depth value.
    :param vmax:                Maximum depth value.
    :param process_state:       Function that preprocesses state.
    :return:                    None.
    """

    print("visualizing state partition")

    for block_idx, block in enumerate(state_partition):

        print("block {:d}".format(block_idx + 1))

        fig, axes = plt.subplots(nrows=num_cells, ncols=num_cells)

        for state_idx, state in enumerate(block):

            if state_idx >= num_cells ** 2:
                break

            if process_state is not None:
                state = process_state(state)

            axis = axes[state_idx // num_cells, state_idx % num_cells]
            im = axis.imshow(state, vmin=vmin, vmax=vmax)
            axis.axis("off")

        # from https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if show:
            plt.show()


def visualize_state_action_partition(state_action_partition, multiplex_action, num_cells=4, vmin=0, vmax=2,
                                     process_state=None, show=True):
    """
    Visualize state-action partition.
    :param state_action_partition:      State-action partition object.
    :param multiplex_action:            Function that paints actions in the depth image.
    :param num_cells:                   Number of cells.
    :param vmin:                        Minimum depth value.
    :param vmax:                        Maximum depth value.
    :param process_state:               Function that processes state.
    :return:                            None.
    """

    print("visualizing state-action partition")

    for block_idx, block in enumerate(state_action_partition.blocks):

        print("block {:d}".format(block_idx + 1))

        fig, axes = plt.subplots(nrows=num_cells, ncols=num_cells)

        for transition_idx, transition in enumerate(block):

            if transition_idx >= num_cells ** 2:
                break

            state = transition.state
            action = transition.action

            if process_state is not None:
                state = process_state(transition.state)

            state = multiplex_action(state, action)

            axis = axes[transition_idx // num_cells, transition_idx % num_cells]
            im = axis.imshow(state, vmin=vmin, vmax=vmax)
            axis.axis("off")

        # from https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if show:
            plt.show()


def multiplex_action(num_moves, num_blocks, block_size, stride, depth, action, paint_value, padding=None,
                     column_first=False):
    """
    Paint an action to the depth image of an environment.
    :param num_moves:       Number of possible positions in the environment.
    :param num_blocks:      Number of blocks the environment is made of.
    :param stride:          Stride of the actions.
    :param depth:           Depth image.
    :param action:          Action as an index.
    :param paint_value:     Depth value to use for multiplexing.
    :return:                Altered depth image.
    """

    action = action % num_moves

    if column_first:
        column = action // num_blocks
        row = action % num_blocks
    else:
        row = action // num_blocks
        column = action % num_blocks

    if padding is None:
        padding = block_size // 2

    depth[int(padding + row * stride) - 2 : int(padding + row * stride) + 2,
          int(padding + column * stride) - 2 : int(padding + column * stride) + 2] = paint_value

    return depth


def multiplex_hand_state(state, paint_value):
    """
    Paint hand state into the depth image.
    :param state:           Tuple of depth and hand state.
    :param paint_value:     Depth value to use for multiplexing.
    :return:                New depth image.
    """

    depth = state[0]
    hand_state = state[1]

    if hand_state > 0:
        depth[:4, :4] = paint_value

    return depth


def unpack_states(states):
    """
    Unpack states into depth images and hand states.
    :param states:      List of states.
    :return:            Arrays of depth images and hand states.
    """

    depths = []
    hand_states = []

    for state in states:
        depths.append(state[0])
        hand_states.append(state[1])

    depths = np.array(depths, dtype=np.float32)
    hand_states = np.array(hand_states, dtype=np.int32)

    return depths, hand_states


def get_colors(num, colormap="Spectral", hex=True):
    """
    Get colors from a matplotlib colormap.
    :param num:             Number of colors.
    :param colormap:        Colormap name.
    :param hex:             Return as hex, otherwise RGBA.
    :return:                List of colors.
    """

    cmap = cm.get_cmap(colormap)
    colors = [cmap(x) for x in np.linspace(0, 1, num=num)]

    if hex:
        colors = ['#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in colors]

    return colors


def smooth(y, box_pts):

    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def image_grid(images, nrows, ncols, vmin, vmax):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    if len(axes.shape) == 1:
        # when there's only one row numpy create an array of shape [num_cols], but my code requires [1, num_cols]
        axes = np.expand_dims(axes, axis=0)

    fig.set_figheight(4 * nrows)
    fig.set_figwidth(4 * ncols)

    for idx, image in enumerate(images):
        axis = axes[idx // ncols, idx % ncols]
        axis.imshow(image, vmin=vmin, vmax=vmax)
        axis.axis("off")

    return fig
