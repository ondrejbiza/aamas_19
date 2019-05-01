import numpy as np


class AbstractModel:

    TEMPLATE_STACK_OF_N_HAND_EMPTY = "stack_of_{:d}_hand_empty"
    TEMPLATE_STACK_OF_N_PUCK_IN_HAND = "stack_of_{:d}_puck_in_hand"
    TEMPLATE_STACK_OF_N_PICK_PUCK = "stack_of_{:d}_pick_puck"
    TEMPLATE_STACK_OF_N_DO_NOTHING = "stack_of_{:d}_do_nothing"
    TEMPLATE_STACK_OF_N_PLACE_PUCK_OR_STACK_OF_N1_DO_NOTHING = "stack_of_{:d}_place_puck_or_stack_of_{:d}_do_nothing"

    def __init__(self, num_pucks):
        """
        Coarsest state-action partition for the task of stacking n pucks.
        :param num_pucks:       Number of pucks to stack.
        """

        self.num_pucks = num_pucks

        self.state_blocks = []
        self.state_action_blocks = []

        for i in range(num_pucks):

            self.state_blocks.append(self.TEMPLATE_STACK_OF_N_HAND_EMPTY.format(i + 1))
            self.state_blocks.append(self.TEMPLATE_STACK_OF_N_PUCK_IN_HAND.format(i + 1))

            if i == 0:
                self.state_action_blocks.append(self.TEMPLATE_STACK_OF_N_DO_NOTHING.format(i + 1))

            if i != num_pucks - 1:
                self.state_action_blocks.append(self.TEMPLATE_STACK_OF_N_PICK_PUCK.format(i + 1))
                self.state_action_blocks.append(
                    self.TEMPLATE_STACK_OF_N_PLACE_PUCK_OR_STACK_OF_N1_DO_NOTHING.format(i + 1, i + 2)
                )

    def get_state_action_block(self, state, next_state):
        """
        Get state-action block from the coarsest partition.
        :param state:           State.
        :param next_state:      Next state.
        :return:                State-action block.
        """

        n = int(np.max(next_state[0]))

        if int(np.max(state[0])) == n - 1:
            return self.TEMPLATE_STACK_OF_N_PLACE_PUCK_OR_STACK_OF_N1_DO_NOTHING.format(n - 1, n)
        elif state[1] == 0 and next_state[1] == 1:
            return self.TEMPLATE_STACK_OF_N_PICK_PUCK.format(n)
        elif n == 1:
            return self.TEMPLATE_STACK_OF_N_DO_NOTHING.format(n)
        else:
            return self.TEMPLATE_STACK_OF_N_PLACE_PUCK_OR_STACK_OF_N1_DO_NOTHING.format(n - 1, n)

    def get_state_block(self, state):
        """
        Get state block from the coarsest partition.
        :param state:   State.
        :return:        State block.
        """

        n = np.max(state[0])

        if state[1] == 0:
            return self.TEMPLATE_STACK_OF_N_HAND_EMPTY.format(n)
        else:
            return self.TEMPLATE_STACK_OF_N_PUCK_IN_HAND.format(n)