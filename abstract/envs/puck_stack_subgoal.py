import numpy as np
from gym import spaces
import matplotlib.pyplot as plt


class PuckStackSubgoal:

    MAX_EPISODE_LENGTH_MULTIPLIER = 5
    NUM_PUCKS = 4

    TASK_STACK_3 = "task_a"
    TASK_STACK_2_AND_2 = "task_b"

    def __init__(self, task, num_blocks=8, block_size=28, show_pick=False, show_place=False, min_radius=8,
                 max_radius=10):

        assert num_blocks >= 2

        self.task = task

        self.show_pick = show_pick
        self.show_place = show_place

        self.blockSize = block_size  # edge size of one block
        self.stride = self.blockSize  # num grid cells between adjacent move destinations
        self.num_blocks = num_blocks

        self.initStride = self.stride
        self.gridSize = self.blockSize * num_blocks
        self.num_pucks = self.NUM_PUCKS

        self.observation_space = spaces.Tuple(
            [spaces.Box(np.zeros([self.gridSize, self.gridSize, 1]), np.ones([self.gridSize, self.gridSize, 1]),
                        dtype=np.float32),
             spaces.Discrete(2)])
        self.holdingImage = []
        self.state = None
        self.max_episode = self.num_pucks * self.MAX_EPISODE_LENGTH_MULTIPLIER
        self.gap = 3  # num pixels that need to be clear around a block in order ot move it.
        #        self.gap = 2 # num pixels that need to be clear around a block in order ot move it.

        self.pucks = []
        for radius in range(min_radius, max_radius):
            X, Y = np.mgrid[0:self.blockSize, 0:self.blockSize]
            halfBlock = int(self.blockSize / 2)
            dist2 = (X - halfBlock) ** 2 + (Y - halfBlock) ** 2
            im = np.int32(dist2 < radius ** 2)
            self.pucks.append(np.reshape(im, int(self.blockSize ** 2)))

        self.reset()

    def reset(self):

        # grid used to select actions
        self.moveCenters = range(int(self.blockSize / 2), int(self.gridSize - (self.blockSize / 2) + 1), self.stride)
        self.num_moves = len(self.moveCenters) ** 2
        self.action_space = spaces.Discrete(2 * self.num_moves)

        # grid on which objects are initially placed
        #        initMoveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),1)
        #        initMoveCenters = range(int(self.blockSize/2),int(self.gridSize-(self.blockSize/2)+1),14)
        initMoveCenters = range(int(self.blockSize / 2), int(self.gridSize - (self.blockSize / 2) + 1), self.initStride)
        #        initMoveCenters = self.moveCenters

        # Initialize state as null
        self.state = []

        halfSide = int(self.blockSize / 2)

        # self.state[0] encodes block layout
        self.state.append(np.zeros(self.observation_space.spaces[0].shape))

        for i in range(self.num_pucks):
            while True:

                ii = initMoveCenters[np.random.randint(len(initMoveCenters))]
                jj = initMoveCenters[np.random.randint(len(initMoveCenters))]
                iiRangeOuter, jjRangeOuter = np.meshgrid(range(ii - halfSide, ii + halfSide),
                                                         range(jj - halfSide, jj + halfSide))

                if True not in (self.state[0][iiRangeOuter, jjRangeOuter, 0] > 0):  # if this block empty
                    blockSel = np.random.randint(np.shape(self.pucks)[0])
                    img = np.reshape(self.pucks[blockSel], [self.blockSize, self.blockSize])
                    self.state[0][iiRangeOuter, jjRangeOuter, 0] = img
                    break

        # self.state[1] encodes what the robot is holding -- start out holding nothing (0)
        self.state.append(0)
        self.episode_timer = 0

        return np.array(self.state)

    def step(self, action):

        X, Y = np.meshgrid(self.moveCenters, self.moveCenters)
        coords = np.stack([np.reshape(Y, [-1]), np.reshape(X, [-1])], axis=0)
        halfSide = int(self.blockSize / 2)

        # if PICK
        if action < self.num_moves:

            # if not holding anything
            if self.state[1] == 0:
                ii = coords[0, action]
                jj = coords[1, action]
                iiRangeInner, jjRangeInner = np.meshgrid(range(ii - halfSide + self.gap, ii + halfSide - self.gap),
                                                         range(jj - halfSide + self.gap, jj + halfSide - self.gap))

                region = self.state[0][iiRangeInner, jjRangeInner, 0]
                max_height = np.max(region)

                if max_height > 0:

                    # get the highest puck
                    self.holdingImage = np.zeros_like(region)
                    self.holdingImage[region == max_height] += 1

                    # subtract it from the region
                    self.state[0][iiRangeInner, jjRangeInner, 0] -= self.holdingImage
                    self.state[1] = 1  # set holding to contents of action target

        # if PLACE
        elif action < 2 * self.num_moves:

            action -= self.num_moves

            if self.state[1] != 0:

                ii = coords[0, action]
                jj = coords[1, action]
                iiRangeInner, jjRangeInner = np.meshgrid(range(ii - halfSide + self.gap, ii + halfSide - self.gap),
                                                         range(jj - halfSide + self.gap, jj + halfSide - self.gap))

                self.state[0][iiRangeInner, jjRangeInner, 0] += np.copy(self.holdingImage)
                self.state[1] = 0  # set holding to zero

        # check for termination condition
        reward = 0
        done = 0

        # check for stack of two blocks
        if (self.task == self.TASK_STACK_3 and self.check_goal_a()) or \
                (self.task == self.TASK_STACK_2_AND_2 and self.check_goal_b()):
            reward = 10
            done = 1

        if self.episode_timer > self.max_episode:
            self.episode_timer = 0
            done = 1
        self.episode_timer += 1

        return self.state, reward, done, {}

    def check_goal_a(self):

        return self.get_max_stack_height() == 3

    def check_goal_b(self):

        num_2_stacks = 0

        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                window = self.state[0][i * self.blockSize : (i + 1) * self.blockSize,
                                       j * self.blockSize : (j + 1) * self.blockSize, :]
                if np.max(window) == 2:
                    num_2_stacks += 1

        return num_2_stacks == 2

    def get_max_stack_height(self):

        return int(np.max(self.state[0]))

    def render(self):
        print("grid:")
        plt.imshow(np.tile(self.state[0], [1, 1, 3]))
        plt.show()

    @staticmethod
    def getBoundingBox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_state(self):
        return self.state
