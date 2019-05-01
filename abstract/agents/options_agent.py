from enum import Enum
import numpy as np
from ..abstraction import utils as abstraction_utils


class OptionsAgent:

    class Exploration(Enum):

        EPSILON_GREEDY = 1
        SOFTMAX = 2

    def __init__(self, quotient_mdp, quotient_mdp_agent, exploration, learning_rate=None, epsilon=0.1,
                 softmax_temperature=1.0, discount=0.9):

        assert isinstance(exploration, self.Exploration)

        self.quotient_mdp = quotient_mdp
        self.quotient_mdp_agent = quotient_mdp_agent
        self.exploration = exploration
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.softmax_temperature = softmax_temperature
        self.discount = discount

        self.q = np.zeros(len(self.quotient_mdp.actions))
        self.c = np.zeros(len(self.quotient_mdp.actions))

    def act(self, observation, timestep):

        return self.quotient_mdp_agent.act(observation, timestep)

    def choose_option(self):

        if self.exploration == self.Exploration.EPSILON_GREEDY:
            option = self.choose_option_epsilon_greedy()
        else:
            option = self.choose_option_softmax()

        self.set_goal(option)
        return option

    def choose_option_epsilon_greedy(self):

        r = np.random.uniform(0, 1)
        if r > self.epsilon:
            action = np.argmax(self.q)
        else:
            action = np.random.randint(0, self.q.shape[0])

        return action

    def choose_option_softmax(self):

        softmax = abstraction_utils.softmax(self.q, self.softmax_temperature)
        action = np.random.choice(range(self.q.shape[0]), p=softmax)

        return action

    def set_goal(self, action):

        # find the state block the action leads to
        goal_state = None

        for key, value in self.quotient_mdp.transitions.items():
            if key[1] == action:
                goal_state = value

        # modify rewards
        for key in self.quotient_mdp.rewards.keys():
            if self.quotient_mdp.transitions[key] == goal_state:
                self.quotient_mdp.rewards[key] = 10.0
            else:
                self.quotient_mdp.rewards[key] = 0.0

        # replay q-values
        self.quotient_mdp.value_iteration()

        # give q-values to the quotient MDP agent
        self.quotient_mdp_agent.q_values = self.quotient_mdp.get_state_action_block_q_values()

    def learn(self, action, episode_rewards):

        return_val = np.sum([(self.discount ** i) * r for i, r in enumerate(episode_rewards)])

        if self.learning_rate is not None:
            self.q[action] += self.learning_rate * (return_val - self.q[action])
        else:
            self.q[action] += abstraction_utils.update_mean(return_val, self.q[action], self.c[action])

        self.c[action] += 1
