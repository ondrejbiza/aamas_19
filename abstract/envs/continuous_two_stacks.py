from .continuous_puck_stack import ContinuousPuckStack


class ContinuousTwoStacks(ContinuousPuckStack):

    def __init__(self, num_blocks, num_pucks, goal, num_actions, min_radius=7, max_radius=12, no_contact=True):

        assert goal * 2 >= num_pucks

        ContinuousPuckStack.__init__(
            self, num_blocks, num_pucks, goal, num_actions, min_radius=min_radius, max_radius=max_radius,
            no_contact=no_contact
        )

        self.max_steps = max(20, 5 * goal * 2)

    def check_goal(self):

        # check if there are two stacks of the target height
        if len(self.layers[self.goal - 1]) != 2:
            return False

        return True
