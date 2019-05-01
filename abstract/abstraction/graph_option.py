import numpy as np
from . import utils


class GraphOption:

    def __init__(self, graph, learning_rate=None):
        """
        Initialize an option agent based on the planning graph.
        :param graph:       Planning graph.
        """

        self.graph = graph
        self.learning_rate = learning_rate

        self.q = np.zeros(len(self.graph.nodes))
        self.c = np.zeros(len(self.graph.nodes))

    def act_softmax(self, temperature):
        """
        Select the goal with softmax exploration.
        :param temperature:     Temperature of the softmax.
        :return:                None.
        """

        softmax = utils.softmax(self.q, temperature)
        action = np.random.choice(range(self.q.shape[0]), p=softmax)

        self.act(action)
        return action

    def act_e_greedy(self, epsilon):
        """
        Select the goal with epsilon greedy exploration.
        :param epsilon:         Epsilon.
        :return:                None.
        """

        r = np.random.uniform(0, 1)
        if r > epsilon:
            action = np.argmax(self.q)
        else:
            action = np.random.randint(0, self.q.shape[0])

        self.act(action)
        return action

    def act(self, action):
        """
        Set a goal for the graph and replan the shortest paths.
        :param action:
        :return:
        """

        self.graph.goal = action
        self.graph.plan()

    def learn(self, action, return_val):
        """
        Update q-value for all actions.
        :param action:          Action index.
        :param return_val:      Return.
        :return:
        """

        if self.learning_rate is not None:
            self.q[action] += self.learning_rate * (return_val - self.q[action])
        else:
            self.q[action] += utils.update_mean(return_val, self.q[action], self.c[action])

        self.c[action] += 1

        print(self.q)
