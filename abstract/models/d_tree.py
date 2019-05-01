import numpy as np
from sklearn.tree import DecisionTreeClassifier


class DTree:

    def __init__(self):

        self.tree = None
        self.trained = False
        self.reset()

    def reset(self):

        self.trained = False
        self.tree = DecisionTreeClassifier()

    def predict(self, states, actions):

        features = self.__encode(states, actions)
        return self.tree.predict_proba(features)

    def fit_split(self, states, actions, labels, mask=None):

        return self.fit(states, actions, labels, mask=mask)

    def fit(self, states, actions, labels, mask=None):

        features = self.__encode(states, actions)
        self.tree.fit(features, labels)
        self.trained = True

        return None, None, None, None

    def __encode(self, states, actions):

        actions_one_hot = np.zeros((len(actions), 16), dtype=np.int32)
        actions_one_hot[np.arange(len(actions)), actions] = 1

        return np.concatenate([states, actions_one_hot], axis=1).astype(np.bool)
