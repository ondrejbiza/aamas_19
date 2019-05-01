import copy as cp
import numpy as np


class QuotientMDPAgent:

    def __init__(self, classifier, q_values, env, fully_convolutional=False, sample_actions=None,
                 proportional_selection=False, softmax_selection=False, softmax_temperature=1.0,
                 random_selection=False):

        self.classifier = classifier
        self.sample_actions = sample_actions
        self.q_values = q_values
        self.env = env
        self.fully_convolutional = fully_convolutional
        self.proportional_selection = proportional_selection
        self.softmax_selection = softmax_selection
        self.softmax_temperature = softmax_temperature
        self.random_selection = random_selection

        self.exploration_schedule = None

    def act(self, state, timestep):

        state = cp.deepcopy(state)
        predictions, actions = self.predict_blocks(state)

        blocks = np.argmax(predictions, axis=1)

        best_block = self.get_best_block(blocks)

        block_actions, block_probabilities = \
            self.get_block_actions_and_probabilities(best_block, blocks, predictions, actions)

        action = self.select_action(block_actions, block_probabilities)

        if state[1] == 1:
            action += predictions.shape[0]

        new_state, reward, done, _ = self.env.step(action)
        new_state = cp.deepcopy(new_state)

        return state, best_block, None, action, new_state, None, None, None, reward, done

    def get_best_block(self, blocks):

        unique_blocks = list(set(blocks))
        q_values = []

        for block in unique_blocks:
            q_value = self.q_values[block]
            q_values.append(q_value)

        best_block = unique_blocks[int(np.argmax(q_values))]

        return best_block

    @staticmethod
    def get_block_actions_and_probabilities(target_block, blocks, predictions, actions):

        block_actions = []
        block_probabilities = []

        for idx, block in enumerate(blocks):
            if block == target_block:
                block_actions.append(actions[idx])
                block_probabilities.append(predictions[idx, block])

        block_probabilities = np.array(block_probabilities, dtype=np.float32)

        return block_actions, block_probabilities

    def select_action(self, block_actions, block_probabilities):

        if self.proportional_selection:
            probabilities = block_probabilities / block_probabilities.sum()
            action = np.random.choice(block_actions, replace=False, p=probabilities)
        elif self.softmax_selection:
            e = np.exp(np.array(block_probabilities, dtype=np.float32) / self.softmax_temperature)
            softmax = e / np.sum(e)
            action = np.random.choice(block_actions, replace=False, p=softmax)
        elif self.random_selection:
            action = np.random.choice(block_actions)
        else:
            action = block_actions[int(np.argmax(block_probabilities))]

        return action

    def predict_blocks(self, state):

        if self.fully_convolutional:
            predictions = self.classifier.predict_all_actions([state])[0]
            actions = list(range(predictions.shape[0]))
        else:
            actions = self.sample_actions(state)
            predictions = self.classifier.predict([state] * len(actions), actions)

        assert len(predictions.shape) == 2

        return predictions, actions
