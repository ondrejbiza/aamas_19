import random
import numpy as np
from .. import constants


class BlocksWorldEnv:

    NUM_POSITIONS = 4
    NUM_BLOCKS = 3
    NUM_COLORS = 4
    STATE_SIZE = NUM_BLOCKS * NUM_POSITIONS + NUM_BLOCKS * NUM_COLORS + NUM_BLOCKS * NUM_BLOCKS
    NUM_ACTIONS = NUM_POSITIONS ** 2
    MAX_STEPS = 20

    STEP_REWARD = -1
    WIN_REWARD = 100

    class Block:

        def __init__(self, color, is_focus):
            """
            Block data structure for the blocks world environment.
            :param color:       Color of the block.
            :param is_focus:    Decides if the block is the focus block.
            """

            self.color = color
            self.is_focus = is_focus

    def __init__(self, encode_state=False, encode_all=False, encode_relational=False, include_colors=False,
                 failure_probability=0.0, difficulty=constants.HARD):
        """
        Blocks world environment.
        :param encode_state:            Encode state for neural nets.
        :param failure_probability:     Probability of failure of any action.
        """

        self.encode_state = encode_state
        self.encode_all = encode_all
        self.encode_relational = encode_relational
        self.include_colors = include_colors
        self.failure_probability = failure_probability
        self.difficulty = difficulty

        self.positions = None
        self.blocks = None
        self.focus_block = None
        self.env_step = None
        self.reset()

        self.target_position = None
        self.target_height = None
        self.reset_goal()

    def reset(self):
        """
        Reset the environment.
        :return:        None.
        """

        # initialize step
        self.env_step = 0

        # initialize positions and blocks array
        self.positions = [[] for _ in range(self.NUM_POSITIONS)]
        self.blocks = []

        # select focus block
        focus_idx = random.randint(0, self.NUM_BLOCKS - 1)

        # place all blocks
        for block_idx in range(self.NUM_BLOCKS):

            color = random.randint(0, self.NUM_COLORS - 1)
            position = random.randint(0, self.NUM_POSITIONS - 1)
            is_focus = block_idx == focus_idx
            block = self.Block(color, is_focus)

            if is_focus:
                self.focus_block = block
            else:
                self.blocks.append(block)

            self.positions[position].append(block)

        # prepend focus block
        self.blocks.insert(0, self.focus_block)

        return self.get_state()

    def reset_goal(self, blacklist=None):
        """
        Reset the goal.
        :param blacklist:       Blacklist of goals (we might not want to use the same goal twice).
        :return:                None.
        """

        while True:
            target_position = np.random.randint(0, self.NUM_POSITIONS)

            if self.difficulty == constants.EASY:
                target_height = np.random.randint(0, 1)
            elif self.difficulty == constants.MEDIUM:
                target_height = np.random.randint(0, 2)
            else:
                target_height = np.random.randint(0, self.NUM_BLOCKS)

            if blacklist is None or (target_position, target_height) not in blacklist:
                break

        self.target_position = target_position
        self.target_height = target_height

    def step(self, action):
        """
        Take an action in the environment.
        :param action:      Action.
        :return:            Reward and is_done.
        """

        self.env_step += 1

        # check if the action is valid
        assert action < self.NUM_ACTIONS

        # de-multiplex action
        take_from = action // self.NUM_POSITIONS
        put_to = action % self.NUM_POSITIONS

        # check if take from and put to equal
        if take_from == put_to:
            return self.get_state(), self.STEP_REWARD, self.env_step >= self.MAX_STEPS, {}

        # check if the action can be executed
        if len(self.positions[take_from]) == 0 or random.uniform(0, 1) < self.failure_probability:
            return self.get_state(), self.STEP_REWARD, self.env_step >= self.MAX_STEPS, {}

        # move the block
        block = self.positions[take_from][-1]
        del self.positions[take_from][-1]
        self.positions[put_to].append(block)

        # check if done
        if self.check_goal():
            return self.get_state(), self.WIN_REWARD, True, {}
        else:
            return self.get_state(), self.STEP_REWARD, self.env_step >= self.MAX_STEPS, {}

    def get_state(self):
        """
        Get state as a vector of one-hot encoded features.
        :return:        Vector of one-hot encoded features.
        """

        if self.encode_state:
            positions = np.zeros((self.NUM_BLOCKS, self.NUM_POSITIONS))
            heights = np.zeros((self.NUM_BLOCKS, self.NUM_BLOCKS))
            colors = np.zeros((self.NUM_BLOCKS, self.NUM_COLORS))

            for position_idx, position in enumerate(self.positions):
                if len(position) > 0:
                    for height_idx, block in enumerate(position):

                        block_idx = self.blocks.index(block)

                        positions[block_idx, position_idx] = 1
                        heights[block_idx, height_idx] = 1
                        colors[block_idx, block.color] = 1

            if self.include_colors:
                features = np.concatenate(
                    [np.reshape(positions, -1), np.reshape(heights, -1), np.reshape(colors, -1)], axis=0
                )
            else:
                features = np.concatenate([np.reshape(positions, -1), np.reshape(heights, -1)], axis=0)

            return features
        elif self.encode_all:

            if self.include_colors:
                enc = np.zeros((self.NUM_POSITIONS, self.NUM_BLOCKS, self.NUM_COLORS), dtype=np.int32)
                focus = np.zeros(self.NUM_POSITIONS + self.NUM_BLOCKS + self.NUM_COLORS)
            else:
                enc = np.zeros((self.NUM_POSITIONS, self.NUM_BLOCKS), dtype=np.int32)
                focus = np.zeros(self.NUM_POSITIONS + self.NUM_BLOCKS)

            for position_idx, position in enumerate(self.positions):
                if len(position) > 0:
                    for height_idx, block in enumerate(position):

                        block_idx = self.blocks.index(block)

                        if self.include_colors:
                            enc[position_idx, height_idx, block.color] = 1
                        else:
                            enc[position_idx, height_idx] = 1

                        if block_idx == 0:
                            focus[position_idx] = 1
                            focus[self.NUM_POSITIONS + height_idx] = 1

                            if self.include_colors:
                                focus[self.NUM_POSITIONS + self.NUM_BLOCKS + block.color] = 1

            features = np.concatenate([focus, np.reshape(enc, -1)], axis=0)
            return features

        elif self.encode_relational:

            if self.include_colors:
                indices_list = [[0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1], [0, 1, 2], [0, 2, 1],
                                [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]]
                focus = np.zeros(self.NUM_POSITIONS + self.NUM_BLOCKS + self.NUM_COLORS, dtype=np.bool)
            else:
                indices_list = [[0], [1], [0, 1], [1, 0]]
                focus = np.zeros(self.NUM_POSITIONS + self.NUM_BLOCKS, dtype=np.bool)

            shapes = [self.NUM_POSITIONS, self.NUM_BLOCKS, self.NUM_COLORS]
            matrices = []
            for indices in indices_list:
                shape = [shapes[i] for i in indices]
                matrices.append(np.zeros(shape, dtype=np.bool))

            for position_idx, position in enumerate(self.positions):
                if len(position) > 0:
                    for height_idx, block in enumerate(position):

                        block_idx = self.blocks.index(block)

                        features = np.array([position_idx, height_idx, block.color])

                        for indices, matrix in zip(indices_list, matrices):
                            if len(indices) == 1:
                                matrix[features[indices][0]] = 1
                            elif len(indices) == 2:
                                matrix[features[indices][0], features[indices][1]] = 1
                            else:
                                matrix[features[indices][0], features[indices][1], features[indices][2]] = 1

                        if block_idx == 0:
                            focus[position_idx] = 1
                            focus[self.NUM_POSITIONS + height_idx] = 1

                            if self.include_colors:
                                focus[self.NUM_POSITIONS + self.NUM_BLOCKS + block.color] = 1

            features = np.concatenate([np.reshape(matrix, -1) for matrix in matrices], axis=0)
            features = np.concatenate([np.reshape(focus, -1), features], axis=0)
            return features

        else:
            positions = np.empty(self.NUM_BLOCKS, dtype=np.int32)
            heights = np.empty(self.NUM_BLOCKS, dtype=np.int32)
            colors = np.empty(self.NUM_BLOCKS, dtype=np.int32)

            for position_idx, position in enumerate(self.positions):
                if len(position) > 0:
                    for height_idx, block in enumerate(position):
                        block_idx = self.blocks.index(block)

                        positions[block_idx] = position_idx
                        heights[block_idx] = height_idx
                        colors[block_idx] = block.color

            if self.include_colors:
                features = np.concatenate([positions, heights, colors], axis=0)
            else:
                features = np.concatenate([positions, heights], axis=0)

            return features

    def check_goal(self):
        """
        Check if the environment is in a goal state.
        :return:        True if goal reached, otherwise False.
        """

        # check that there is some block in the target position
        if len(self.positions[self.target_position]) <= self.target_height:
            return False

        # check that the block is the focus block
        if not self.positions[self.target_position][self.target_height].is_focus:
            return False

        return True
