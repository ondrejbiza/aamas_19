import numpy as np
from .schedules import LinearSchedule


class GGraphAgent:

    def __init__(self, g_model, sample_actions, graph, env, softmax_selection=False, softmax_temperature=1.0,
                 random_selection=False, alternative_policy=None, confidence_exp=False, eps_start=1.0, eps_end=0.1,
                 exp_timesteps=None, exp_softmax=False):
        """
        Initialize an agent based on a state-action partition and a planning graph.
        :param g_model:                     State-action block classifier.
        :param sample_actions:              Function that samples actions for each state.
        :param graph:                       Planning graph.
        :param env:                         Environment to act in.
        :param softmax_selection:           Select actions corresponding to the same block with softmax.
        :param softmax_temperature:         Temperature of the softmax function.
        :param random_selection:            Select actions corresponding to the same block randomly.
        :param alternative_policy:          An alternative policy to use if we do not know which way to go.
        """

        assert not (softmax_selection and random_selection)

        self.model = g_model
        self.sample_actions = sample_actions
        self.graph = graph
        self.env = env
        self.softmax_selection = softmax_selection
        self.softmax_temperature = softmax_temperature
        self.random_selection = random_selection
        self.alternative_policy = alternative_policy
        self.confidence_exp = confidence_exp
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.exp_timesteps = exp_timesteps
        self.exp_softmax = exp_softmax

        self.exploration_schedule = None
        if self.confidence_exp:
            self.__setup_exploration()

    def act(self, state, timestep):
        """
        Take a step in the environment.
        :param state:       Current state.
        :param timestep:    Current timestep.
        :return:
        """
        actions = self.sample_actions(state)

        predictions = self.model.predict([state] * len(actions), actions)
        blocks = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        min_block_length = None
        if self.confidence_exp and np.random.uniform(0, 1) < self.exploration_schedule.value(timestep):
            # select the action we are the least confident about
            rec_confidences = (1 - confidences)
            if self.exp_softmax:
                probs = np.exp(rec_confidences) / np.sum(np.exp(rec_confidences))
            else:
                probs = rec_confidences / np.sum(rec_confidences)

            action = np.random.choice(list(range(len(confidences))), p=probs)
        else:
            unique_blocks = list(set(blocks))
            unique_blocks_length = []

            for block in unique_blocks:

                length = self.graph.get_length(block)

                if length is not None:
                    unique_blocks_length.append(length)

            if len(unique_blocks_length) > 0:
                # path from the current state to goal found, find out which action we should take
                min_block = unique_blocks[int(np.argmin(unique_blocks_length))]
                min_block_length = np.min(unique_blocks_length)
                block_actions = []
                block_probs = []

                for idx, block in enumerate(blocks):

                    if block == min_block:

                        block_actions.append(actions[idx])
                        block_probs.append(predictions[idx, block])

                if self.softmax_selection:
                    e = np.exp(np.array(block_probs, dtype=np.float32) / self.softmax_temperature)
                    softmax = e / np.sum(e)

                    try:
                        # choose an action in proportion to the probabilities
                        action = np.random.choice(block_actions, replace=False, p=softmax)
                    except ValueError:
                        # some weird edge case, I don't know why this happens
                        action = np.random.choice(block_actions)

                elif self.random_selection:
                    action = np.random.choice(block_actions)
                else:
                    action = block_actions[int(np.argmax(block_probs))]
            else:
                # we don't know which way to go--use the policy that collect the initial data if specified
                # or a random policy instead
                if self.alternative_policy is not None:
                    return self.alternative_policy.act(state, timestep)
                else:
                    action = np.random.choice(actions)

        # check if we are using the same actions for pick and place
        if state[1] == 1 and action < self.env.num_blocks ** 2:
            action += self.env.num_blocks ** 2

        new_state, reward, done, _ = self.env.step(action)

        return state, min_block_length, None, action, new_state, None, None, None, reward, done

    def __setup_exploration(self):
        """
        Setup the exploration schedule.
        :return:        None.
        """

        assert self.confidence_exp and self.exp_timesteps is not None

        self.exploration_schedule = LinearSchedule(
            schedule_timesteps=self.exp_timesteps, initial_p=self.eps_start, final_p=self.eps_end
        )


class GMDPAgent:

    def __init__(self, g_model, sample_actions, q_values, env, softmax_selection=False, softmax_temperature=1.0,
                 random_selection=False, alternative_policy=None):
        """
        Initialize an agent based on a state-action partition and q-values of individual state-action blocks.
        :param g_model:                     State-action block classifier.
        :param sample_actions:              Function that samples actions for each state.
        :param q_values:                    Dictionary of q-values for each state-action block.
        :param env:                         Environment to act in.
        :param softmax_selection:           Select actions corresponding to the same block with softmax.
        :param softmax_temperature:         Temperature of the softmax function.
        :param random_selection:            Select actions corresponding to the same block randomly.
        :param alternative_policy:          An alternative policy to use if we do not know which way to go.
        """

        assert not (softmax_selection and random_selection)

        self.model = g_model
        self.sample_actions = sample_actions
        self.q_values = q_values
        self.env = env
        self.softmax_selection = softmax_selection
        self.softmax_temperature = softmax_temperature
        self.random_selection = random_selection
        self.alternative_policy = alternative_policy

    def act(self, state, timestep):
        """
        Take a step in the environment.
        :param state:       Current state.
        :param timestep:    Current timestep.
        :return:
        """

        actions = self.sample_actions(state)

        predictions = self.model.predict([state] * len(actions), actions)
        blocks = np.argmax(predictions, axis=1)

        unique_blocks = list(set(blocks))
        current_q_values = []

        for block in unique_blocks:
            q_value = self.q_values[block]
            current_q_values.append(q_value)

        if len(current_q_values) > 0:

            best_block = unique_blocks[int(np.argmax(current_q_values))]
            block_actions = []
            block_probs = []

            for idx, block in enumerate(blocks):
                if block == best_block:
                    block_actions.append(actions[idx])
                    block_probs.append(predictions[idx, block])

            if self.softmax_selection:
                e = np.exp(np.array(block_probs, dtype=np.float32) / self.softmax_temperature)
                softmax = e / np.sum(e)
                action = np.random.choice(block_actions, replace=False, p=softmax)
            elif self.random_selection:
                action = np.random.choice(block_actions)
            else:
                action = block_actions[int(np.argmax(block_probs))]
        else:
            # we don't know which way to go--use the policy that collect the initial data if specified
            # or a random policy instead
            if self.alternative_policy is not None:
                return self.alternative_policy.act(state, timestep)
            else:
                action = np.random.choice(actions)

        new_state, reward, done, _ = self.env.step(action)

        return state, None, None, action, new_state, None, None, None, reward, done
