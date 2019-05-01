import copy as cp
import numpy as np
import tensorflow as tf
from . import utils as agent_utils
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .schedules import LinearSchedule
from . import tf_util_rob as U


class DQNAgent:

    SCOPE = "deep_q"
    FUNCTION = "q_func"
    TARGET_FUNCTION = "target_q_func"

    def __init__(self, env, num_states, num_actions, neural_net_func, optimizer, exploration_fraction, init_explore,
                 max_timesteps, exploration_final_epsilon, batch_size, target_network=True, target_update_freq=100,
                 grad_norm_clipping=1.0, buffer_size=10000, depth_mean=1, prioritized_replay=False, pr_alpha=0.6,
                 pr_beta0=0.4, pr_eps=1e-6, gamma=0.9, custom_graph=False):
        """
        Create a gamma function agent.
        :param env:                             Instance of an environment.
        :param num_states:                      Number of states.
        :param neural_net_func:                 Function that build a neural network.
        :param optimizer:                       An instance of Tensorflow optimizer.
        :param grad_norm_clipping:              Clip gradient norm to this value.
                                                If None, use stride from the environment.
        :param depth_mean:                      Mean to subtract from the depth image.
        """

        assert num_actions % num_states == 0

        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.neural_net_func = neural_net_func
        self.optimizer = optimizer
        self.grad_norm_clipping = grad_norm_clipping
        self.batch_size = batch_size

        self.target_network = target_network
        self.target_update_freq = target_update_freq

        self.exploration_fraction = exploration_fraction
        self.init_explore = init_explore
        self.max_timesteps = max_timesteps
        self.exploration_final_epsilon = exploration_final_epsilon

        self.buffer_size = buffer_size
        self.depth_mean = depth_mean

        self.prioritized_replay = prioritized_replay
        self.pr_alpha = pr_alpha
        self.pr_beta0 = pr_beta0
        self.pr_eps = pr_eps
        self.gamma = gamma

        self.custom_graph = custom_graph
        self.tf_graph = tf.get_default_graph()
        if custom_graph:
            self.tf_graph = tf.Graph()

        self.make_obs_ph = None
        self.make_actions_ph = None
        self.make_target_ph = None
        self.make_weight_ph = None
        self.make_abstract_action_ph = None

        with self.tf_graph.as_default():

            self.__build_make_placeholders()

            self.get_q_list = None
            self.__build_networks()

            self.train_q_list = None
            self.__build_training()

            self.exploration_schedule = None
            self.__setup_exploration()

            self.replay_buffer = None
            self.__setup_replay_buffer()

        self.session = None

    def start_session(self, num_cpu, gpu_memory_fraction):
        """
        Start and initialize a session.
        :param num_cpu:                     Number of CPUs to use.
        :param gpu_memory_fraction:         Fraction of GPu memory to use.
        :return:                            None.
        """

        with self.tf_graph.as_default():
            self.session = U.make_session(num_cpu, gpu_memory_fraction=gpu_memory_fraction)

            with self.session.as_default():
                U.initialize()

    def stop_session(self):
        """
        Stop the session.
        :return:            None.
        """

        self.session.close()

    def act(self, observation, timestep):
        """
        Take a single step in the environment.
        :param observation:         An observation of teh current state.
        :param timestep:            The current time step (from the start of the training, sets the value of epsilon).
        :return:                    Current and next state of the environment with other useful information.
        """

        with self.tf_graph.as_default():
            with self.session.as_default():

                # get gamma values
                hand_state = observation[1]
                q_values = self.get_q(np.expand_dims(observation[0], axis=0), [hand_state])[0]

                if hand_state == 0:
                    q_values = q_values[:self.num_actions // 2]
                else:
                    q_values = q_values[self.num_actions // 2:]

                q_values_noise = q_values + np.random.random(np.shape(q_values)) * 0.01
                action = np.argmax(q_values_noise)

                if self.exploration_schedule is not None and np.random.rand() < \
                        self.exploration_schedule.value(timestep):
                    action = np.random.randint(self.num_actions // 2)

                if hand_state == 1:
                    action += self.num_actions // 2

                # copy observation
                observation = cp.deepcopy(observation)

                # execute action
                new_observation, reward, done, _ = self.env.step(action)

                # remember experience
                self.replay_buffer.add(
                    observation, action, reward, cp.deepcopy(new_observation), done
                )

                return observation, None, None, action, new_observation, None, None, None, reward, done

    def remember(self, state, action, reward, next_state, done):

        self.replay_buffer.add(
            state, action, reward, next_state, done
        )

    def learn(self, timestep):
        """
        Take a single training step.
        :return:        None.
        """

        with self.tf_graph.as_default():
            with self.session.as_default():

                # get batch
                if self.prioritized_replay:
                    beta = self.beta_schedule.value(timestep)
                    states, actions, rewards, next_states, dones, weights, batch_indexes = \
                        self.replay_buffer.sample(self.batch_size, beta)
                else:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    weights, batch_indexes = np.ones_like(rewards), None

                # train
                depth_images = np.stack(states[:, 0], axis=0)
                hand_states = np.stack(states[:, 1], axis=0)

                next_depth_images = np.stack(next_states[:, 0], axis=0)
                next_hand_states = np.stack(next_states[:, 1], axis=0)

                # calculate targets
                a = rewards
                b = (1 - dones) * self.gamma
                c = np.empty(next_hand_states.shape[0])

                m1 = next_hand_states == 0
                m2 = next_hand_states == 1

                if self.target_network:
                    c[m1] = np.max(self.get_target_q(next_depth_images, next_hand_states)[m1, :self.num_actions // 2],
                                   axis=1)
                    c[m2] = np.max(self.get_target_q(next_depth_images, next_hand_states)[m2, self.num_actions // 2:],
                                   axis=1)
                else:
                    c[m1] = np.max(self.get_q(next_depth_images, next_hand_states)[m1, :self.num_actions // 2], axis=1)
                    c[m2] = np.max(self.get_q(next_depth_images, next_hand_states)[m2, self.num_actions // 2:], axis=1)

                targets = a + b * c

                # train
                td_error = self.train_q(
                    depth_images, hand_states, actions, targets, weights
                )

                new_priorities = np.abs(td_error) + self.pr_eps

                if self.prioritized_replay:
                    self.replay_buffer.update_priorities(batch_indexes, new_priorities)

        # maybe update target network
        if self.target_network and timestep % self.target_update_freq == 0 and timestep > 0:
            self.session.run(self.target_update_op)

    def update_exploration(self, fraction, final_epsilon):
        """
        Update exploration schedule.
        :param fraction:            Fraction of timesteps to explore.
        :param final_epsilon:       Final epsilon after the end of exploration.
        :return:                    None.
        """

        self.exploration_fraction = fraction
        self.exploration_final_epsilon = final_epsilon
        self.__setup_exploration()

    def save(self, path):
        """
        Save Tensorflow model.
        :param path:        Save path, the saver will create three files.
        :return:            None.
        """

        with self.tf_graph.as_default():
            with self.session.as_default():
                saver = tf.train.Saver()
                saver.save(self.session, path)

    def load(self, path):
        """
        Load Tensorflow model.
        :param path:        Load path, name of the model without suffix (three files should share the name).
        :return:            None.
        """

        with self.tf_graph.as_default():
            with self.session.as_default():
                saver = tf.train.Saver()
                saver.restore(self.session, path)

    def empty_replay_buffer(self):
        """
        Empty the replay buffer.
        :return:        None.
        """

        self.__setup_replay_buffer()

    def delete_graph(self):

        self.tf_graph = None

    def __build_make_placeholders(self):
        """
        Define lambdas that make placeholders.
        :return:        None.
        """

        self.make_obs_ph = lambda name: U.BatchInput(self.env.observation_space.spaces[0].shape, name=name)
        self.make_hand_ph = lambda name: U.BatchInput([], name=name, dtype=tf.int32)
        self.make_actions_ph = lambda name: U.BatchInput([], name=name, dtype=tf.int32)
        self.make_target_ph = lambda name: U.BatchInput([], name=name)
        self.make_weight_ph = lambda name: U.BatchInput([], name=name)

    def __build_networks(self):
        """
        Build a neural network for each state.
        :return:        None.
        """

        self.get_q = agent_utils.build_getq_branch(
            depth_ph=self.make_obs_ph,
            hand_ph=self.make_hand_ph,
            q_func=self.neural_net_func,
            scope=self.SCOPE,
            qscope=self.FUNCTION,
            num_actions=self.num_actions
        )

        self.get_target_q = None
        if self.target_network:

            self.get_target_q = agent_utils.build_getq_branch(
                depth_ph=self.make_obs_ph,
                hand_ph=self.make_hand_ph,
                q_func=self.neural_net_func,
                scope=self.SCOPE,
                qscope=self.TARGET_FUNCTION,
                num_actions=self.num_actions
            )

    def __build_training(self):
        """
        Define training functions for each neural network from build_networks.
        :return:        None.
        """

        self.train_q = agent_utils.build_q_branch_target_train(
            make_depths_ph=self.make_obs_ph,
            make_hand_ph=self.make_hand_ph,
            make_actions_ph=self.make_actions_ph,
            make_target_ph=self.make_target_ph,
            make_weight_ph=self.make_weight_ph,
            num_actions=self.num_actions,
            q_func=self.neural_net_func,
            optimizer=self.optimizer,
            grad_norm_clipping=self.grad_norm_clipping,
            scope=self.SCOPE,
            qscope=self.FUNCTION
        )

        if self.target_network:
            self.__build_target_update()

    def __build_target_update(self):

        assert self.target_network

        source_scope = "{}/{}".format(self.SCOPE, self.FUNCTION)
        target_scope = "{}/{}".format(self.SCOPE, self.TARGET_FUNCTION)

        source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source_scope)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)

        assert len(source_vars) == len(target_vars) and len(source_vars) > 0

        update_ops = []
        for source_var, target_var in zip(source_vars, target_vars):
            update_ops.append(tf.assign(target_var, source_var))

        self.target_update_op = tf.group(*update_ops)

    def __setup_exploration(self):
        """
        Setup the exploration schedule.
        :return:        None.
        """

        self.exploration_schedule = None
        if self.exploration_fraction > 0:
            self.exploration_schedule = LinearSchedule(
                schedule_timesteps=int(self.exploration_fraction * self.max_timesteps), initial_p=self.init_explore,
                final_p=self.exploration_final_epsilon
            )

    def __setup_replay_buffer(self):
        """
        Setup a replay buffer.
        :return:        None.
        """

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.pr_alpha)
            self.beta_schedule = LinearSchedule(self.max_timesteps, initial_p=self.pr_beta0, final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None
