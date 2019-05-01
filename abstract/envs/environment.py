import numpy as np


class Puck:

    def __init__(self, block_size, radius, allow_stacking=False):
        """
        Puck object.
        :param block_size:      Size of a block in the grid world environment.
        :param allow_stacking:  Allow stacking of pucks.
        :param radius:          Radius of the puck.
        """

        self.block_size = block_size
        self.radius = radius
        self.allow_stacking = allow_stacking
        self.depth = 1

        self.mask = None
        self.draw()

    def draw(self):
        """
        Create a mask for the puck.
        :return:
        """

        x, y = np.mgrid[0 : self.block_size, 0:self.block_size]
        half_block = int(self.block_size / 2)
        dist2 = (x - half_block) ** 2 + (y - half_block) ** 2
        self.mask = np.bool_(dist2 < self.radius ** 2)

    def can_place(self, grid, i, j):
        """
        Check if the puck can be placed in a virtual grid.
        :param grid:    Virtual grid.
        :param i:       Coordinate 1.
        :param j:       Coordinate 2.
        :return:        True if can be placed, otherwise False.
        """

        if self.allow_stacking:
            return True
        else:
            return len(grid.grid[i][j]) == 0


class Bridge:

    HORIZONTAL = 1
    VERTICAL = 2

    PART_MIDDLE = 1
    PART_END_TOP_OR_LEFT = 2
    PART_END_RIGHT_OR_BOTTOM = 3

    class Part:

        def __init__(self, block_size, orientation, type, bridge):
            """
            Part of a bridge.
            :param block_size:      Block size in a grid world environment.
            :param orientation:     Orientation of the bridge.
            :param type:            Type of the part.
            :param bridge:          Reference to the bridge object.
            """

            self.block_size = block_size
            self.orientation = orientation
            self.type = type
            self.bridge = bridge
            self.depth = 1

            self.mask = None
            self.draw()

        def draw(self):
            """
            Draw a mask for the part.
            :return:        None.
            """

            self.mask = np.zeros((self.block_size, self.block_size), dtype=np.bool)

            if self.orientation == Bridge.HORIZONTAL:
                if self.type == Bridge.PART_MIDDLE:
                    self.mask[int((1 / 4) * self.block_size) : int((3 / 4) * self.block_size), :] = True
                elif self.type == Bridge.PART_END_TOP_OR_LEFT:
                    self.mask[int((1 / 4) * self.block_size) : int((3 / 4) * self.block_size),
                              int((1 / 2) * self.block_size) : ] = True
                else:
                    self.mask[int((1 / 4) * self.block_size): int((3 / 4) * self.block_size),
                              : int((1 / 2) * self.block_size)] = True
            else:
                if self.type == Bridge.PART_MIDDLE:
                    self.mask[:, int((1 / 4) * self.block_size) : int((3 / 4) * self.block_size)] = True
                elif self.type == Bridge.PART_END_TOP_OR_LEFT:
                    self.mask[int((1 / 2) * self.block_size) :,
                              int((1 / 4) * self.block_size): int((3 / 4) * self.block_size)] = True
                else:
                    self.mask[: int((1 / 2) * self.block_size),
                              int((1 / 4) * self.block_size): int((3 / 4) * self.block_size)] = True

    def __init__(self, block_size, orientation):
        """
        A bridge object.
        :param block_size:          Block size in a grid world environment.
        :param orientation:         Orientation of the bridge.
        """

        assert orientation in [self.HORIZONTAL, self.VERTICAL]

        self.block_size = block_size
        self.orientation = orientation

        self.parts = None
        self.construct()

    def construct(self):
        """
        Create all parts of the bridge.
        :return:        None.
        """

        self.parts = []
        self.parts.append(self.Part(self.block_size, self.orientation, self.PART_END_TOP_OR_LEFT, self))
        self.parts.append(self.Part(self.block_size, self.orientation, self.PART_MIDDLE, self))
        self.parts.append(self.Part(self.block_size, self.orientation, self.PART_END_RIGHT_OR_BOTTOM, self))

    def add_to_grid(self, grid, i, j):
        """
        Add the bridge to a virtual grid.
        :param grid:        Virtual grid.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :return:            None.
        """

        if self.orientation == self.HORIZONTAL:
            self.set_depth(min(len(grid.grid[i][j - 1]) + 1, 2))
            grid.add(self.parts[0], i, j - 1)
            grid.add(self.parts[1], i, j)
            grid.add(self.parts[2], i, j + 1)
        else:
            self.set_depth(min(len(grid.grid[i - 1][j]) + 1, 2))
            grid.add(self.parts[0], i - 1, j)
            grid.add(self.parts[1], i, j)
            grid.add(self.parts[2], i + 1, j)

    def remove_from_grid(self, grid, i, j):
        """
        Remove the bridge from a virtual grid.
        :param grid:        Virtual grid.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :return:            None.
        """

        coordinates_list = []

        if self.orientation == self.HORIZONTAL:
            coordinates_list.append((i, j - 1))
            coordinates_list.append((i, j))
            coordinates_list.append((i, j + 1))
        else:
            coordinates_list.append((i - 1, j))
            coordinates_list.append((i, j))
            coordinates_list.append((i + 1, j))

        for coordinates in coordinates_list:
            cell = grid.grid[coordinates[0]][coordinates[1]]
            for idx, obj in enumerate(cell):
                if isinstance(obj, self.Part) and obj.bridge == self:
                    grid.remove(coordinates[0], coordinates[1], idx)
                    break

    def can_place(self, grid, i, j):
        """
        Check if the bridge can be placed.
        :param grid:        Virtual grid.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :return:            True if can be placed, otherwise False.
        """

        return self.can_place_on_ground(grid, i, j) or self.can_place_on_pucks(grid, i, j)

    def can_place_on_ground(self, grid, i, j):
        """
        Check if the bridge can be placed on the ground.
        :param grid:        Virtual grid.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :return:            True / False.
        """

        h = grid.height
        w = grid.width
        g = grid.grid

        if self.orientation == self.HORIZONTAL:
            if j - 1 >= 0 and j + 1 < w and (len(g[i][j - 1]) == len(g[i][j]) == len(g[i][j + 1]) == 0):
                # region free
                return True
        else:
            if i - 1 >= 0 and i + 1 < h and (len(g[i - 1][j]) == len(g[i][j]) == len(g[i + 1][j]) == 0):
                # region free
                return True

        return False

    def can_place_on_pucks(self, grid, i, j):
        """
        Check if the bridge can be placed on pucks.
        :param grid:        Virtual grid.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :return:            True / False.
        """

        h = grid.height
        w = grid.width
        g = grid.grid

        if self.orientation == self.HORIZONTAL:
            if j - 1 >= 0 and j + 1 < w:
                if len(g[i][j - 1]) == len(g[i][j + 1]) == 1 and isinstance(g[i][j + 1][0], Puck) and \
                   isinstance(g[i][j + 1][0], Puck) and len(g[i][j]) == 0:
                    # puck at each end
                    return True
                elif len(g[i][j - 1]) >= 1 and isinstance(g[i][j - 1][0], Puck) and len(g[i][j + 1]) >= 1 and isinstance(
                        g[i][j + 1][0], Puck) and (len(g[i][j - 1]) == 1 or isinstance(g[i][j - 1][1], self.Part)) \
                        and (len(g[i][j - 1]) == 1 or isinstance(g[i][j - 1][1], self.Part)):
                    # puck at each end + other bridge end on the puck
                    return True
        else:
            if i - 1 >= 0 and i + 1 < h:
                if len(g[i - 1][j]) == len(g[i + 1][j]) == 1 and isinstance(g[i + 1][j][0], Puck) and \
                   isinstance(g[i + 1][j][0], Puck) and len(g[i][j]) == 0:
                    # puck at each end
                    return True
                elif len(g[i - 1][j]) >= 1 and isinstance(g[i - 1][j][0], Puck) and len(g[i+ 1][j]) >= 1 and isinstance(
                     g[i + 1][j][0], Puck) and (len(g[i - 1][j]) == 1 or isinstance(g[i - 1][j][1], self.Part)) and \
                     (len(g[i - 1][j]) == 1 or isinstance(g[i - 1][j][1], self.Part)):
                    # puck at each end + other bridge end on the puck
                    return True

        return False

    def set_depth(self, depth):
        """
        Set depth of all parts of the bridge.
        :param depth:   Depth.
        :return:        None.
        """

        for part in self.parts:
            part.depth = depth


class VirtualGrid:

    def __init__(self, block_size, height, width, height_noise_std=0.0):
        """
        Virtual grid for the grid world environment.
        :param block_size:      Block size.
        :param height:          Height of the grid.
        :param width:           Width of the grid.
        """

        self.block_size = block_size
        self.height = height
        self.width = width
        self.height_noise_std = height_noise_std

        self.grid = None
        self.image = None

        self.reset()

    def reset(self):
        """
        Reset the grid.
        :return:        None.
        """

        self.grid = [[[] for _ in range(self.width)] for _ in range(self.height)]
        self.image = np.zeros((self.block_size * self.height, self.block_size * self.width), dtype=np.float32)

    def add(self, obj, i, j, render=True):
        """
        Add an object to the grid.
        :param obj:         Object to add.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :param render:      Render the object and add it to the depth map.
        :return:            None.
        """

        self.grid[i][j].append(obj)

        if render:
            self.render_block(i, j)

    def remove(self, i, j, k, render=True):
        """
        Remove an object from the grid.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :param k:           Index of the object.
        :param render:      Render the object and add it to the depth map.
        :return:            None.
        """

        self.grid[i][j].pop(k)

        if render:
            self.render_block(i, j)

    def remove_last(self, i, j, render=True):
        """
        Remove the last added object from the grid.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :param render:      Render the object and add it to the depth map.
        :return:            None.
        """

        self.grid[i][j].pop()

        if render:
            self.render_block(i, j)

    def render_block(self, i, j):
        """
        Render a single block.
        :param i:           Coordinate 1.
        :param j:           Coordinate 2.
        :return:            None.
        """

        block_range = np.index_exp[i * self.block_size : (i + 1) * self.block_size,
                                   j * self.block_size : (j + 1) * self.block_size]

        self.image[block_range] = 0

        for idx, obj in enumerate(self.grid[i][j]):

            self.image[block_range][obj.mask] = np.random.normal(
                obj.depth, self.height_noise_std, size=self.image[block_range].shape
            )[obj.mask]
