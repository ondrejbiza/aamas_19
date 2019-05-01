import numpy as np


class Transition:

    def __init__(self, state, action, reward, next_state, is_start, is_end, state_block=None, next_state_block=None,
                 state_action_block=None, next_state_block_confidence=None, task=None):

        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_start = is_start
        self.is_end = is_end

        self.state_block = state_block
        self.next_state_block = next_state_block
        self.state_action_block = state_action_block
        self.next_state_block_confidence = next_state_block_confidence
        self.task = task

    def simulate_next_state_block_confidence(self, num_blocks, block_indices, predictions, num_sims=1000):
        """
        Simulate next state block confidence given a probability distribution over all state-action blocks
        for each state-action pair.
        :param num_blocks:          Number of state-action blocks.
        :param block_indices:       State-action block indices constituting the target state block.
        :param predictions:         Predictions for each state-action pair.
        :param num_sims:            Number of simulations to run.
        :return:                    Simulated confidence.
        """

        block_indices = set(block_indices)

        sims = []
        for prediction in predictions:
            sim = np.random.choice(list(range(num_blocks)), size=num_sims, replace=True, p=prediction)
            sims.append(sim)
        sims = np.stack(sims, axis=0)

        mask = np.apply_along_axis(lambda x: set(x) == block_indices, 0, sims)

        self.next_state_block_confidence = np.sum(mask) / num_sims
