import random
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from . import environment


class BridgesEnv:

    MIN_PUCK_RADIUS = 8
    MAX_PUCK_RADIUS = 11

    HAND_EMPTY = 0
    PUCK_IN_HAND = 1
    BRIDGE_IN_HAND = 2
    HAND_FULL = 1

    def __init__(self, num_pucks=2, num_bridges=1, block_size=28, num_blocks=8, only_horizontal=True,
                 binary_state=False, max_episode_length=None, height_noise_std=0.0):
        """
        Create a block world environment with pucks and bridges.
        :param num_pucks:           Number of pucks in the environment.
        :param num_bridges:         Number of bridges in the environment.
        :param block_size:          Sizes of blocks.
        :param only_horizontal:     Place only horizontal bridges.
        :param binary_state:        Holding / not holding; otherwise not holding, holding puck and holding bridge.
        :param num_blocks:          Number of blocks (height and width of the environment).
        """

        self.num_pucks = num_pucks
        self.num_bridges = num_bridges
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.only_horizontal = only_horizontal
        self.binary_state = binary_state
        self.height_noise_std = height_noise_std

        self.num_positions = self.num_blocks ** 2
        self.grid_size = block_size * num_blocks

        self.max_episode_length = max_episode_length
        if self.max_episode_length is None:
            self.max_episode_length = (num_pucks + num_bridges) * 5

        self.observation_space = spaces.Tuple(
            [spaces.Box(np.zeros([self.grid_size, self.grid_size, 1]), np.ones([self.grid_size, self.grid_size, 1]),
                        dtype=np.float32),
             spaces.Discrete(2)])

        self.grid = environment.VirtualGrid(
            self.block_size, self.num_blocks, self.num_blocks, height_noise_std=height_noise_std
        )
        self.hand_state = None
        self.in_hand = None
        self.pucks = None
        self.bridges = None
        self.episode_timer = None

        self.reset()

    def reset(self):
        """
        Reset the environment: place all pucks and bridges.
        :return:
        """

        # reset virtual grid
        self.grid.reset()
        self.pucks = []
        self.bridges = []
        self.episode_timer = 0

        # reset hand
        self.hand_state = self.HAND_EMPTY
        self.in_hand = None

        # get all free coordinates
        free_coordinates = set()
        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                free_coordinates.add((i, j))

        # place pucks
        for i in range(self.num_pucks):

            puck = environment.Puck(self.block_size, random.randint(self.MIN_PUCK_RADIUS, self.MAX_PUCK_RADIUS))
            self.pucks.append(puck)
            coordinates = random.sample(free_coordinates, 1)[0]
            free_coordinates.remove(coordinates)

            self.grid.add(puck, coordinates[0], coordinates[1])

        # place bridges
        for i in range(self.num_bridges):

            if self.only_horizontal:
                orientation = environment.Bridge.HORIZONTAL
            else:
                orientation = random.choice([environment.Bridge.VERTICAL, environment.Bridge.HORIZONTAL])

            bridge = environment.Bridge(self.block_size, orientation)
            self.bridges.append(bridge)

            while True:

                middle = random.sample(free_coordinates, 1)[0]

                if bridge.can_place_on_ground(self.grid, middle[0], middle[1]):
                    free_coordinates.remove(middle)
                    bridge.add_to_grid(self.grid, middle[0], middle[1])
                    break

        return self.get_state()

    def step(self, action):
        """
        Take a single step in the environment.
        :param action:      Action to execute.
        :return:            State after taking the action.
        """

        if action < self.num_positions:

            # pick
            x = action // self.num_blocks
            y = action % self.num_blocks

            if self.hand_state == self.HAND_EMPTY:

                # pick
                if len(self.grid.grid[x][y]) > 0:

                    obj = self.grid.grid[x][y][-1]

                    if isinstance(obj, environment.Puck):

                        # pick puck
                        self.in_hand = obj
                        if self.binary_state:
                            self.hand_state = self.HAND_FULL
                        else:
                            self.hand_state = self.PUCK_IN_HAND
                        self.grid.remove_last(x, y)

                    else:

                        # pick bridge
                        if obj.type == environment.Bridge.PART_MIDDLE:
                            # can only pick from the middle
                            bridge = obj.bridge
                            self.in_hand = bridge
                            if self.binary_state:
                                self.hand_state = self.HAND_FULL
                            else:
                                self.hand_state = self.BRIDGE_IN_HAND
                            bridge.remove_from_grid(self.grid, x, y)

        elif action < 2 * self.num_positions:

            # place
            action -= self.num_positions

            x = action // self.num_blocks
            y = action % self.num_blocks

            if isinstance(self.in_hand, environment.Puck):

                # place puck
                if self.in_hand.can_place(self.grid, x, y):
                    self.grid.add(self.in_hand, x, y)
                    self.hand_state = self.HAND_EMPTY
                    self.in_hand = None

            elif isinstance(self.in_hand, environment.Bridge):

                # place bridge
                if self.in_hand.can_place(self.grid, x, y):
                    self.in_hand.add_to_grid(self.grid, x, y)
                    self.hand_state = self.HAND_EMPTY
                    self.in_hand = None

        # check for termination condition
        reward = 0
        done = 0

        # check for stack of two blocks
        if np.all([bridge.parts[1].depth == 2 for bridge in self.bridges]):
            reward = 10
            done = 1

        if self.episode_timer > self.max_episode_length:
            self.episode_timer = 0
            done = 1

        self.episode_timer += 1

        return self.get_state(), reward, done, {}

    def show(self):
        """
        Show the current depth image.
        :return:        None.
        """

        plt.imshow(self.grid.image)
        plt.colorbar()
        plt.axis("off")
        plt.show()

    def get_state(self):
        return [np.expand_dims(self.grid.image, axis=2), self.hand_state]