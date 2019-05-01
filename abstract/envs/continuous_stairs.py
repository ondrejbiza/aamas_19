from .continuous_puck_stack import ContinuousPuckStack


class ContinuousStairs(ContinuousPuckStack):

    def __init__(self, num_blocks, num_pucks, goal, num_actions, min_radius=7, max_radius=12, tolerance=8,
                 no_contact=True, height_noise_std=0.05):

        assert goal in [3, 6]

        self.tolerance = tolerance

        ContinuousPuckStack.__init__(
            self, num_blocks, num_pucks, goal, num_actions, min_radius=min_radius, max_radius=max_radius,
            no_contact=no_contact, height_noise_std=height_noise_std
        )

    def check_goal(self):

        if self.goal == 3:

            # check if there's a stack of 2
            if len(self.layers[1]) != 1:
                return False

            top_puck = self.layers[1][0]

            # check if the third puck is close to the stack of 2
            if len(self.layers[0]) != 2:
                return False

            if self.layers[0][0].top is None:
                third_puck = self.layers[0][0]
            elif self.layers[0][1].top is None:
                third_puck = self.layers[0][1]
            else:
                raise ValueError("The third puck is obstructed; shouldn't happen.")

            dist = self.euclid(top_puck, [third_puck.x, third_puck.y])

            if dist > top_puck.radius + third_puck.radius + self.tolerance:
                return False

            return True

        else:

            # check if there's a stack of 3
            if len(self.layers[2]) != 1:
                return False

            # check if there's a stack of 2
            if len(self.layers[1]) != 2:
                return False

            # check if the sixth puck is on the ground
            if len(self.layers[0]) != 3:
                return False

            # check if the stack of 2 is close to the stack of 3
            top_puck = self.layers[2][0]

            if self.layers[1][0].top is None:
                second_top_puck = self.layers[1][0]
            elif self.layers[1][1].top is None:
                second_top_puck = self.layers[1][1]
            else:
                raise ValueError("The second top puck is obstructed; shouldn't happen.")

            dist = self.euclid(top_puck, [second_top_puck.x, second_top_puck.y])

            if dist > top_puck.radius + second_top_puck.radius + self.tolerance:
                return False

            # check if the sixth puck is close to the stack of 2
            if self.layers[0][0].top is None:
                sixth_puck = self.layers[0][0]
            elif self.layers[0][1].top is None:
                sixth_puck = self.layers[0][1]
            elif self.layers[0][2].top is None:
                sixth_puck = self.layers[0][2]
            else:
                raise ValueError("The sixth puck is obstructed; shouldn't happen.")

            dist = self.euclid(second_top_puck, [sixth_puck.x, sixth_puck.y])

            if dist > second_top_puck.radius + sixth_puck.radius + self.tolerance:
                return False

            return True
