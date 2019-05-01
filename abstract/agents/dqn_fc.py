import copy as cp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from . import utils as agent_utils
from .. import constants
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .schedules import LinearSchedule
from . import tf_util_rob as U


class DQNFC:

    MOMENTUM = 0.9
    Q_FUNC = "q_func"
    TARGET_Q_FUNC = "target_q_func"

    def __init__(self, env, state_shape, output_shape, num_filters_list, filter_size_list, stride_list, learning_rate,
                 batch_size, optimizer, exploration_fraction, init_explore, final_explore, max_timesteps, dilation=None,
                 target_net=True, target_update_freq=100, grad_norm_clipping=None, buffer_size=10000, depth_mean=1,
                 prioritized_replay=False, pr_alpha=0.6, pr_beta=0.4, pr_eps=1e-6, gamma=0.9,
                 final_filter_size=1, upsample=True, custom_graph=False, show_q_values=False, show_q_values_offset=0,
                 upsample_before=False, upsample_after=False, deconv_before=False, deconv_after=False,
                 deconv_filter_size=5, deconv_before_num_filters=32, end_filter_sizes=None,
                 double_q_network=False, use_memory=False):

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
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.exploration_fraction = exploration_fraction
        self.init_explore = init_explore
        self.final_explore = final_explore
        self.max_timesteps = max_timesteps
        self.dilation = dilation
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
        self.final_filter_size = final_filter_size
        self.upsample = upsample
        self.custom_graph = custom_graph
        self.show_q_values = show_q_values
        self.show_q_values_offset = show_q_values_offset
        self.upsample_before = upsample_before
        self.upsample_after = upsample_after
        self.deconv_before = deconv_before
        self.deconv_after = deconv_after
        self.deconv_filter_size = deconv_filter_size
        self.deconv_before_num_filters = deconv_before_num_filters
        self.end_filter_sizes = end_filter_sizes
        self.double_q_network = double_q_network
        self.use_memory = use_memory

        if self.use_memory:
            self.reset_memory()

        self.tf_graph = tf.get_default_graph()
        if custom_graph:
            self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():

            self.__build_placeholders()

            self.__build_networks()

            if self.target_net:
                # throws error if called after build training because there aren't any momentum variables
                # for the target net
                # feel free to fix this
                self.__build_target_update()

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

                # get Q-values
                observation = cp.deepcopy(observation)
                tmp_observation = self.maybe_add_memory(observation)

                q_values = self.session.run(self.get_q, feed_dict={
                    self.depth_pl: np.expand_dims(tmp_observation[0], axis=0),
                    self.hand_states_pl: [tmp_observation[1]]
                })[0]

                if self.show_q_values:
                    if timestep > self.show_q_values_offset:
                        q_values_reshape = np.reshape(q_values, (self.env.num_actions, self.env.num_actions))
                        plt.subplot(1, 2, 1)
                        plt.imshow(q_values_reshape)
                        plt.subplot(1, 2, 2)
                        plt.imshow(tmp_observation[0][:, :, 0])
                        plt.pause(0.1)
                        plt.clf()

                q_values_noise = q_values + np.random.random(np.shape(q_values)) * 0.01
                action = np.argmax(q_values_noise)

                if self.exploration_schedule is not None and np.random.rand() < \
                        self.exploration_schedule.value(timestep):
                    action = np.random.randint(self.num_actions // 2)

                env_action = action
                if tmp_observation[1] == 1:
                    env_action += self.num_actions // 2

                # execute action
                new_observation, reward, done, _ = self.env.step(env_action)

                # maybe update memory
                if self.use_memory:
                    if tmp_observation[1] == 0 and new_observation[1] == 1:
                        # pick => save the state into the memory
                        self.memory = new_observation[0]
                    elif tmp_observation[1] == 1 and new_observation[1] == 0:
                        # place => reset memory
                        self.reset_memory()

                new_observation = cp.deepcopy(new_observation)

                # remember experience
                self.replay_buffer.add(
                    tmp_observation, action, reward, self.maybe_add_memory(new_observation), done
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

                # maybe use target network
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

                # maybe use double Q-learning
                if self.double_q_network:
                    assert self.target_net

                    tmp_qs = self.session.run(self.get_q, feed_dict={
                        self.depth_pl: next_depth_images,
                        self.hand_states_pl: next_hand_states
                    })
                    argmax_qs = np.argmax(tmp_qs, axis=1)
                    c = qs[list(range(len(argmax_qs))), argmax_qs]
                else:
                    c = np.max(qs, axis=1)

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

    def __build_placeholders(self):

        if self.use_memory:
            state_shape = (self.state_shape[0], self.state_shape[1], 2 * self.state_shape[2])
        else:
            state_shape = self.state_shape

        self.depth_pl = tf.placeholder(tf.float32, shape=(None, *state_shape), name="depth_pl")
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

                        if self.dilation is not None:
                            tmp_dilation = (self.dilation[i], self.dilation[i])
                        else:
                            tmp_dilation = (1, 1)

                        x = tf.layers.conv2d(
                            x, self.num_filters_list[i], self.filter_size_list[i], self.stride_list[i],
                            padding="SAME", activation=tf.nn.relu,
                            kernel_initializer=agent_utils.get_mrsa_initializer(),
                            dilation_rate=tmp_dilation
                        )

            # upsample the output or throw an error
            if x.shape[1].value != self.output_shape[0] or x.shape[2].value != self.output_shape[1]:

                if self.upsample_before:

                    with tf.variable_scope("upsample_before"):
                        x = tf.image.resize_bilinear(x, (self.output_shape[0], self.output_shape[1]), name="upsample")

                elif self.deconv_before:

                    with tf.variable_scope("deconv_before"):
                        ratio = self.output_shape[0] // x.shape[1].value
                        x = tf.layers.conv2d_transpose(
                            x, self.deconv_before_num_filters, self.deconv_filter_size,
                            strides=(ratio, ratio), padding="SAME", activation=tf.nn.relu,
                            kernel_initializer=agent_utils.get_mrsa_initializer()
                        )

            with tf.variable_scope("logits"):

                # predict for both hand empty and hand full
                logits = tf.layers.conv2d(
                    x, 2, self.final_filter_size, 1, padding="SAME", activation=None,
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                if logits.shape[1] != self.output_shape[0] or logits.shape[2] != self.output_shape[1]:

                    if self.upsample_after:

                        with tf.variable_scope("upsample_after"):
                            logits = tf.image.resize_bilinear(
                                logits, (self.output_shape[0], self.output_shape[1]), name="upsample_logits"
                            )

                    elif self.deconv_after:

                        with tf.variable_scope("deconv_after"):

                            ratio = self.output_shape[0] // x.shape[1].value

                            with tf.variable_scope("hand_empty"):
                                logits_1 = tf.layers.conv2d_transpose(
                                    logits[:, :, :, :1], 1, self.deconv_filter_size,
                                    strides=(ratio, ratio), padding="SAME",
                                    kernel_initializer=agent_utils.get_mrsa_initializer()
                                )
                                if self.end_filter_sizes is not None:
                                    for idx, filter_size in enumerate(self.end_filter_sizes):
                                        logits_1 = tf.nn.relu(logits_1)
                                        with tf.variable_scope("end_conv{:d}".format(idx + 1)):
                                            logits_1 = tf.layers.conv2d(
                                                logits_1, 1, kernel_size=filter_size, strides=(1, 1), padding="SAME",
                                                activation=None, kernel_initializer=agent_utils.get_mrsa_initializer()
                                            )

                            with tf.variable_scope("hand_full"):
                                logits_2 = tf.layers.conv2d_transpose(
                                    logits[:, :, :, 1:2], 1, self.deconv_filter_size,
                                    strides=(ratio, ratio), padding="SAME",
                                    kernel_initializer=agent_utils.get_mrsa_initializer()
                                )
                                if self.end_filter_sizes is not None:
                                    for idx, filter_size in enumerate(self.end_filter_sizes):
                                        logits_2 = tf.nn.relu(logits_2)
                                        with tf.variable_scope("end_conv{:d}".format(idx + 1)):
                                            logits_2 = tf.layers.conv2d(
                                                logits_2, 1, kernel_size=filter_size, strides=(1, 1), padding="SAME",
                                                activation=None, kernel_initializer=agent_utils.get_mrsa_initializer()
                                            )

                            logits = tf.concat([logits_1, logits_2], axis=3)

                # mask hand states
                assert logits.shape[-1].value == 2

                mask = tf.cast(self.hand_states_pl, tf.bool)
                logits = tf.where(
                    mask, x=logits[:, :, :, :1], y=logits[:, :, :, 1:]
                )

                assert logits.shape[1].value == self.output_shape[0]
                assert logits.shape[2].value == self.output_shape[1]

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
            self.beta_schedule = LinearSchedule(self.max_timesteps, initial_p=self.pr_beta, final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None

    def reset_memory(self):

        self.memory = np.zeros(self.state_shape, dtype=np.float32)

    def maybe_add_memory(self, state):

        if self.use_memory:
            return [np.concatenate([state[0], self.memory], axis=-1), state[1]]
        else:
            return state
