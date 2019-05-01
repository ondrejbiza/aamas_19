import copy as cp
import numpy as np
import tensorflow as tf
from . import utils as agent_utils
from .. import constants
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .schedules import LinearSchedule
from . import tf_util_rob as U

class DQN:

    MOMENTUM = 0.9
    Q_FUNC = "q_func"
    TARGET_Q_FUNC = "target_q_func"

    def __init__(self, env, state_shape, output_shape, num_filters_list, filter_size_list, stride_list, hiddens, learning_rate,
                 batch_size, optimizer, exploration_fraction, init_explore, final_explore, max_timesteps,
                 target_net=True, target_update_freq=100, grad_norm_clipping=None, buffer_size=10000, depth_mean=1,
                 prioritized_replay=False, pr_alpha=0.6, pr_beta=0.4, pr_eps=1e-6, gamma=0.9, custom_graph=False):

        assert optimizer in [constants.OPT_ADAM, constants.OPT_SGD, constants.OPT_MOMENTUM]
        assert len(state_shape) == 3
        assert len(output_shape) == 2

        self.env = env
        self.state_shape = state_shape
        self.output_shape = output_shape
        self.num_actions = 2 * (output_shape[0] * output_shape[1])
        self.num_filters_list = num_filters_list
        self.filter_size_list = filter_size_list
        self.stride_list = stride_list
        self.hiddens = hiddens
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.exploration_fraction = exploration_fraction
        self.init_explore = init_explore
        self.final_explore = final_explore
        self.max_timesteps = max_timesteps
        self.target_net = target_net
        self.target_update_freq = target_update_freq
        self.grad_norm_clipping = grad_norm_clipping
        self.buffer_size = buffer_size
        self.depth_mean = depth_mean
        self.prioritized_replay = prioritized_replay
        self.pr_alpha = pr_alpha
        self.pr_beta = pr_beta
        self.pr_eps = pr_eps
        self.gamma = gamma
        self.custom_graph = custom_graph

        self.tf_graph = tf.get_default_graph()
        if custom_graph:
            self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():

            self.__build_placeholders()

            self.get_q_list = None
            self.__build_networks()

            if self.target_net:
                # throws error if called after build training because there aren't any momentum variables
                # for the target net
                # feel free to fix this
                self.__build_target_update()

            self.train_q_list = None
            self.__build_training()

            self.exploration_schedule = None
            self.__setup_exploration()

            self.replay_buffer = None
            self.__setup_replay_buffer()

        self.session = None

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

                q_values = self.session.run(self.get_q, feed_dict={
                    self.depth_pl: np.expand_dims(observation[0], axis=0),
                    self.hand_states_pl: [hand_state]
                })[0]

                q_values_noise = q_values + np.random.random(np.shape(q_values)) * 0.01
                action = np.argmax(q_values_noise)

                if self.exploration_schedule is not None and np.random.rand() < \
                        self.exploration_schedule.value(timestep):
                    action = np.random.randint(self.num_actions // 2)

                env_action = action
                if hand_state == 1:
                    env_action += self.num_actions // 2

                # copy observation
                observation = cp.deepcopy(observation)

                # execute action
                new_observation, reward, done, _ = self.env.step(env_action)

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

                if self.target_net:
                    qs = self.session.run(self.get_target_q, feed_dict={
                        self.depth_pl: next_depth_images,
                        self.hand_states_pl: next_hand_states
                    })
                else:
                    qs = self.session.run(self.get_q, feed_dict={
                        self.depth_pl: next_depth_images,
                        self.hand_states_pl: next_hand_states
                    })

                c[m1] = np.max(qs[m1], axis=1)
                c[m2] = np.max(qs[m2], axis=1)

                targets = a + b * c

                # train
                td_error, _ = self.session.run([self.td_error, self.train_step], feed_dict={
                    self.depth_pl: depth_images,
                    self.hand_states_pl: hand_states,
                    self.actions_pl: actions,
                    self.targets_pl: targets,
                    self.importance_weights_pl: weights
                })

                new_priorities = np.abs(td_error) + self.pr_eps

                if self.prioritized_replay:
                    self.replay_buffer.update_priorities(batch_indexes, new_priorities)

        # maybe update target network
        if self.target_net and timestep % self.target_update_freq == 0 and timestep > 0:
            self.session.run(self.target_update_op)

    def mask_hand_states(self, logits, hand_states_pl):
        """
        Mask hand states and aggregate logits.
        :param logits:              Logits Tensor.
        :param hand_states_pl:      Hand states placeholder.
        :return:                    Aggregated logits Tensor.
        """

        shape = [x.value for x in logits.shape]
        shape[0] = -1
        shape[3] = 2
        shape.append(1)

        logits = tf.reshape(logits, shape=shape)
        logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])

        # mask hand states
        shape[3] = 1
        shape[4] = 2
        mask = tf.one_hot(hand_states_pl, 2, dtype=tf.float32)
        for s in shape[1:4]:
            mask = tf.stack([mask for _ in range(s)], axis=-2)

        logits_mask = logits * mask
        return tf.reduce_sum(logits_mask, axis=4)

    def update_exploration(self, fraction, final_epsilon):
        """
        Update exploration schedule.
        :param fraction:            Fraction of timesteps to explore.
        :param final_epsilon:       Final epsilon after the end of exploration.
        :return:                    None.
        """

        self.exploration_fraction = fraction
        self.final_explore = final_epsilon
        self.__setup_exploration()

    def empty_replay_buffer(self):
        """
        Empty the replay buffer.
        :return:        None.
        """

        self.__setup_replay_buffer()

    def start_session(self, num_cpu=None, gpu_memory_fraction=None):
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

    def __build_placeholders(self):

        self.depth_pl = tf.placeholder(tf.float32, shape=(None, *self.state_shape), name="depth_pl")
        self.hand_states_pl = tf.placeholder(tf.int32, shape=(None,), name="hand_states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="labels_pl")
        self.targets_pl = tf.placeholder(tf.float32, shape=(None,), name="targets_pl")
        self.importance_weights_pl = tf.placeholder(tf.float32, shape=(None,), name="importance_weights_pl")

    def __build_networks(self):

        self.get_q = self.__build_network(self.Q_FUNC)
        self.get_target_q = self.__build_network(self.TARGET_Q_FUNC)

    def __build_network(self, namespace):

        with tf.variable_scope(namespace):

            x = self.depth_pl

            with tf.variable_scope("convs"):

                for i in range(len(self.num_filters_list)):

                    with tf.variable_scope("conv{:d}".format(i + 1)):

                        x = tf.layers.conv2d(
                            x, self.num_filters_list[i], self.filter_size_list[i], self.stride_list[i],
                            padding="SAME", activation=tf.nn.relu,
                            kernel_initializer=agent_utils.get_mrsa_initializer()
                        )

            x = tf.layers.flatten(x, name="flatten")

            with tf.variable_scope("fcs"):

                for i in range(len(self.hiddens)):

                    with tf.variable_scope("fc{:d}".format(i + 1)):

                        x = tf.layers.dense(
                            x, self.hiddens[i], activation=tf.nn.relu
                        )

            with tf.variable_scope("logits"):

                # predict for both hand empty and hand full
                logits = tf.layers.dense(x, self.output_shape[0] * self.output_shape[1] * 2)
                logits = tf.reshape(logits, shape=(-1, self.output_shape[0], self.output_shape[1], 1))

                # mask hand states
                logits = self.mask_hand_states(logits, self.hand_states_pl)

                return tf.reshape(
                    logits, shape=(-1, self.output_shape[0] * self.output_shape[1])
                )

    def __build_training(self):

        mask = tf.one_hot(self.actions_pl, self.num_actions // 2, dtype=tf.float32)
        qs = tf.reduce_sum(self.get_q * mask, axis=1)

        self.td_error = qs - tf.stop_gradient(self.targets_pl)
        self.errors = self.importance_weights_pl * U.huber_loss(self.td_error)

        if self.optimizer == constants.OPT_ADAM:
            opt = tf.train.AdamOptimizer(self.learning_rate, self.MOMENTUM)
        elif self.optimizer == constants.OPT_MOMENTUM:
            opt = tf.train.MomentumOptimizer(self.learning_rate, self.MOMENTUM)
        else:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        if self.grad_norm_clipping is not None:
            # clip gradients to some norm
            gradients = opt.compute_gradients(self.errors)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)
            self.train_step = opt.apply_gradients(gradients)
        else:
            self.train_step = opt.minimize(
                self.errors
            )

    def __build_target_update(self):

        source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.Q_FUNC)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.TARGET_Q_FUNC)

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
                final_p=self.final_explore
            )

    def __setup_replay_buffer(self):
        """
        Setup a replay buffer.
        :return:        None.
        """

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.pr_alpha)
            self.beta_schedule = LinearSchedule(self.max_timesteps, initial_p=self.pr_beta, final_p=self.final_explore)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None

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
