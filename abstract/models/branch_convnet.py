import time
import collections
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from ..agents import utils as agent_utils
from . import utils as balancing_utils
from ..tf_calibrate import calibration


class BranchConvNet:

    def __init__(self, state_shape, num_actions, num_hand_states, num_filters_list, filter_size_list, stride_list,
                 hiddens_embed, hiddens_final, learning_rate, batch_size, weight_decay, momentum=0.9,
                 max_training_steps=5000, validation_frequency=100, g_mean=False, gpu_memory_fraction=0.1,
                 exp_lr_decay=False, lr_decay_rate=0.96, lr_decay_step=1000, preprocess=None, set_best_params=True,
                 validation_fraction=0.2, balance=None, verbose=False, fixed_num_classes=None, fit_reset=True,
                 max_retraining_steps=None, early_stop_no_improvement_steps=None, calibrate=False,
                 normalize=False, gradient_clip_norm=None, mrsa_conv_init=False, mrsa_fc_init=False,
                 batch_norm=False):
        """
        A ConvNet with one branch for images and one branch for hand states and actions.
        :param state_shape:                         Shape of the state.
        :param num_actions:                         Number of actions.
        :param num_hand_states:                     Number of hand states.
        :param hiddens_embed:                       Hidden layers in the embedding branch.
        :param hiddens_final:                       Hidden layers after branch concatenation.
        :param learning_rate:                       Learning rate.
        :param batch_size:                          Batch size.
        :param weight_decay:                        Weight decay, applied to all parameterized layers.
        :param validation_frequency:                How often to perform validation.
        :param momentum:                            Momentum for SGD optimizer.
        :param g_mean:                              Use geometric mean during validation.
        :param verbose:                             Print additional information.
        :param early_stop_no_improvement_steps:     Stop training if there is no improvement for this many steps.
        """

        # cannot calibrate without a validation set
        assert not ((validation_fraction is None or validation_fraction == .0) and calibrate)

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_hand_states = num_hand_states
        self.hiddens_embed = hiddens_embed
        self.hiddens_final = hiddens_final
        self.num_filters_list = num_filters_list
        self.filter_size_list = filter_size_list
        self.stride_list = stride_list
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_training_steps = max_training_steps
        self.validation_frequency = validation_frequency
        self.g_mean = g_mean
        self.gpu_memory_fraction = gpu_memory_fraction
        self.exp_lr_decay = exp_lr_decay
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.preprocess = preprocess
        self.set_best_params = set_best_params
        self.validation_fraction = validation_fraction
        self.balance = balance
        self.verbose = verbose
        self.fixed_num_classes = fixed_num_classes
        self.fit_reset = fit_reset
        self.max_retraining_steps = max_retraining_steps
        self.early_stop_no_improvement_steps = early_stop_no_improvement_steps
        self.calibrate = calibrate
        self.gradient_clip_norm = gradient_clip_norm
        self.mrsa_conv_init = mrsa_conv_init
        self.mrsa_fc_init = mrsa_fc_init
        self.batch_norm = batch_norm

        if self.fixed_num_classes is not None:
            self.num_classes = self.fixed_num_classes
        else:
            self.num_classes = None

        self.total_num_training_steps = 0

        self.session = None
        self.global_step = None
        self.learning_rate_schedule = None
        self.mask = None

        # calibration variables
        self.temperature = None
        self.scaled_logits = None
        self.scaled_predictions = None

        # normalization
        self.normalize = normalize
        self.depth_means = None

    def predict(self, states, actions):
        """
        Predict labels for given data.
        :param states:          States
        :param actions:         Actions.
        :return:                Predictions.
        """

        # preprocess data
        if self.preprocess is not None:
            depths, hand_states = self.preprocess(states)
        else:
            depths = states[0]
            hand_states = states[1]

        # maybe normalize the depths
        if self.normalize and self.depth_means is not None:
            depths = depths - self.depth_means

        # generate predictions
        if self.calibrate and self.scaled_predictions is not None:
            predictions_t = self.scaled_predictions
        else:
            predictions_t = self.predictions

        predictions = self.session.run(predictions_t, feed_dict={
            self.depth_pl: depths,
            self.hand_state_pl: hand_states,
            self.actions_pl: actions,
            self.mask_pl: self.mask,
            self.is_training_pl: False
        })

        return predictions

    def fit_split(self, states, actions, labels, mask=None):

        if self.validation_fraction is None or self.validation_fraction == 0.0:

            # maybe balance the dataset
            if self.balance is not None:

                indices = np.array(list(range(len(states))), dtype=np.int32)

                x, labels = self.balance(indices, labels)
                states = states[x]
                actions = actions[x]

                if self.verbose:
                    print("after balancing:")
                    for class_idx in range(int(np.max(labels)) + 1):
                        print("class {:d}: {:d} transitions".format(
                            class_idx + 1, int(np.sum(labels == class_idx)))
                        )
                    print()

                assert states.shape[0] == actions.shape[0] == labels.shape[0]

            return self.fit(states, actions, labels, None, None, None, mask=mask)
        else:
            # create training and validation splits
            indices = np.array(list(range(len(states))), dtype=np.int32)

            train_x, train_labels, valid_x, valid_labels = balancing_utils.stratified_split(
                indices, labels, self.validation_fraction
            )

            train_states = states[train_x]
            train_actions = actions[train_x]
            valid_states = states[valid_x]
            valid_actions = actions[valid_x]

            assert train_states.shape[0] == train_actions.shape[0] == train_labels.shape[0]
            assert valid_states.shape[0] == valid_actions.shape[0] == valid_labels.shape[0]

            # maybe balance the dataset
            if self.balance is not None:

                indices = np.array(list(range(len(train_states))), dtype=np.int32)

                train_x, train_labels = self.balance(indices, train_labels)
                train_states = train_states[train_x]
                train_actions = train_actions[train_x]

                if self.verbose:
                    print("after balancing:")
                    for class_idx in range(int(np.max(train_labels)) + 1):
                        print("class {:d}: {:d} transitions".format(
                            class_idx + 1, int(np.sum(train_labels == class_idx)))
                        )
                    print()

                assert train_states.shape[0] == train_actions.shape[0] == train_labels.shape[0]

            return self.fit(train_states, train_actions, train_labels, valid_states, valid_actions, valid_labels,
                            mask=mask)

    def fit(self, train_states, train_actions, train_labels, valid_states, valid_actions, valid_labels, mask=None):
        """
        Fit the ConvNet to the given data.
        :param train_states:            Training states.
        :param train_actions:           Training actions.
        :param train_labels:            Training labels.
        :param valid_states:            Validation states.
        :param valid_actions:           Validation actions.
        :param valid_labels:            Validation labels.
        :param mask:                    Logits mask.
        :return:                        Best balanced and unbalanced accuracy, best per-class accuracy, best step.
        """

        # preprocess data
        if self.preprocess is not None:
            train_depths, train_hand_states = self.preprocess(train_states)

            if valid_states is not None:
                valid_depths, valid_hand_states = self.preprocess(valid_states)
            else:
                valid_depths, valid_hand_states = None, None
        else:
            train_depths = train_states[0]
            train_hand_states = train_states[1]

            if valid_states is not None:
                valid_depths = valid_states[0]
                valid_hand_states = valid_states[1]
            else:
                valid_depths = None
                valid_hand_states = None

        # maybe compute number of classes
        if self.fixed_num_classes is None:
            self.num_classes = np.max(train_labels) + 1

        # maybe create a mask for logits
        if mask is None:
            self.mask = np.ones(self.num_classes, dtype=np.bool)
        else:
            self.mask = mask

        # check if the neural network was trained before
        if self.session is None:
            first_run = True
        else:
            first_run = False

        # maybe calculate the depth means and normalize the data
        if self.normalize:
            if first_run:
                self.depth_means = np.mean(train_depths, axis=0)

            train_depths = train_depths - self.depth_means
            valid_depths = valid_depths - self.depth_means

        # maybe reset the session and rebuild the neural network
        if self.fit_reset or first_run:
            # reset
            self.reset()

            # build placeholders
            self.__build_placeholders()

            # maybe build learning rate schedule
            if self.exp_lr_decay:
                self.__build_lr_schedule()

            # build network
            self.__build_network()
            self.__build_training()

            # start session
            self.__start_session()

        # train the network
        best_balanced_accuracy, best_unbalanced_accuracy, best_class_accuracies, best_step, best_parameters = \
            self.train(
                train_depths, train_hand_states, train_actions, train_labels, valid_depths, valid_hand_states,
                valid_actions, valid_labels, first_run=first_run, validate=valid_states is not None
            )

        # maybe set the best parameters
        if valid_states is not None and self.set_best_params:

            if self.verbose:
                print("best balanced accuracy: {:.2f}%".format(best_balanced_accuracy * 100))

            for idx, variable in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
                self.session.run(variable.assign(best_parameters[idx]))

        # maybe calibrate confidences
        if self.calibrate:

            logits = self.__predict_logits(valid_depths, valid_hand_states, valid_actions, valid_labels)

            self.temperature = calibration.temperature_scale(logits, self.session, valid_labels)
            self.scaled_logits = self.logits / self.temperature
            self.scaled_predictions = tf.nn.softmax(self.scaled_logits)

        return best_balanced_accuracy, best_unbalanced_accuracy, best_class_accuracies, best_step

    def train(self, train_depths, train_hand_states, train_actions, train_labels, valid_depths, valid_hand_states,
              valid_actions, valid_labels, first_run=False, validate=True):
        """
        Train the neural network.
        :param train_depths:            Training depth images.
        :param train_hand_states:       Training hand states.
        :param train_actions:           Training actions.
        :param train_labels:            Training labels.
        :param valid_depths:            Validation depth images.
        :param valid_hand_states:       Validation hand states.
        :param valid_actions:           Validation actions.
        :param valid_labels:            Validation labels.
        :param first_run:               Is this the first training run.
        :param validate:                Compute validation accuracy. If False, valid_depths, valid_hand_states,
                                        valid_actions and valid_labels can be None.
        :return:                        Best balanced, unbalanced and per-class accuracy, best step and best parameters.
        """

        # train network
        num_steps_per_epoch = train_depths.shape[0] // self.batch_size
        best_balanced_accuracy = None
        best_unbalanced_accuracy = None
        best_class_accuracies = None
        best_step = None
        best_parameters = None

        if not first_run and self.max_retraining_steps is not None:
            max_training_steps = self.max_retraining_steps
        else:
            max_training_steps = self.max_training_steps

        losses = collections.defaultdict(list)
        epoch_start_time = time.time()

        for step_idx in range(max_training_steps):

            epoch_step_idx = step_idx % num_steps_per_epoch

            if epoch_step_idx == 0:
                # new epoch, reshuffle dataset
                train_depths, train_hand_states, train_actions, train_labels = shuffle(
                    train_depths, train_hand_states, train_actions, train_labels
                )
            # build feed dictionary
            batch_slice = np.index_exp[epoch_step_idx * self.batch_size:(epoch_step_idx + 1) * self.batch_size]

            feed_dict = {
                self.depth_pl: train_depths[batch_slice],
                self.hand_state_pl: train_hand_states[batch_slice],
                self.actions_pl: train_actions[batch_slice],
                self.labels_pl: train_labels[batch_slice],
                self.mask_pl: self.mask,
                self.is_training_pl: True
            }

            # take training step
            _, loss, cross_entropy, regularization = self.session.run(
                [self.train_step, self.loss, self.cross_entropy, self.regularization], feed_dict=feed_dict
            )

            losses["total"].append(loss)
            losses["cross_entropy"].append(cross_entropy)
            losses["regularization"].append(regularization)

            if validate and step_idx > 0 and step_idx % self.validation_frequency == 0:

                epoch_duration = time.time() - epoch_start_time
                valid_start_time = time.time()

                # perform validation
                if self.verbose:
                    for key, value in losses.items():
                        print("mean {} loss: {:.8f}".format(key, np.mean(value)))

                    print("validation, step {:d}".format(step_idx))

                losses = collections.defaultdict(list)

                balanced_accuracy, unbalanced_accuracy, class_accuracies = self.validate(
                    valid_depths, valid_hand_states, valid_actions, valid_labels
                )

                if best_balanced_accuracy is None or balanced_accuracy > best_balanced_accuracy:

                    best_balanced_accuracy = balanced_accuracy
                    best_unbalanced_accuracy = unbalanced_accuracy
                    best_class_accuracies = class_accuracies
                    best_step = step_idx

                    if self.set_best_params:
                        best_parameters = self.session.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

                valid_duration = time.time() - valid_start_time
                if self.verbose:
                    print("epoch duration: {:.2f} seconds".format(epoch_duration))
                    print("validation duration: {:.2f} seconds".format(valid_duration))
                epoch_start_time = time.time()

            elif not validate and step_idx > 0 and step_idx % 100 == 0:

                # print only the training loss
                if self.verbose:
                    print("step {:d}".format(step_idx))

                    for key, value in losses.items():
                        print("mean {} loss: {:.8f}".format(key, np.mean(value)))

                losses = collections.defaultdict(list)

            # maybe stop the training early
            if validate and (best_step is not None) and \
                    (self.early_stop_no_improvement_steps is not None) and \
                    (step_idx - best_step >= self.early_stop_no_improvement_steps):
                break

        self.total_num_training_steps += self.max_training_steps

        return best_balanced_accuracy, best_unbalanced_accuracy, best_class_accuracies, best_step, best_parameters

    def validate(self, valid_depths, valid_hand_states, valid_actions, valid_labels):
        """
        Perform validation.
        :param valid_depths:            Validation depth images.
        :param valid_hand_states:       Validation hand states.
        :param valid_actions:           Validation actions.
        :param valid_labels:            Validation labels.
        :return:                        Balanced validation accuracy, unbalanced accuracy and accuracy for each class.
        """

        logits = self.__predict_logits(valid_depths, valid_hand_states, valid_actions, valid_labels)

        class_accuracies = []
        for class_idx in range(self.num_classes):
            if self.mask[class_idx]:
                class_accuracy = np.mean(
                    (np.argmax(logits[valid_labels == class_idx], axis=1) == class_idx).astype(np.float32))
                class_accuracies.append(class_accuracy)

                if self.verbose:
                    print("class {:d}: {:.2f}% accuracy".format(class_idx + 1, class_accuracy * 100))

        if self.g_mean:
            balanced_accuracy = np.power(np.product(class_accuracies), 1 / len(class_accuracies))
        else:
            balanced_accuracy = np.mean(class_accuracies)

        unbalanced_accuracy = np.mean((np.argmax(logits, axis=1) == valid_labels).astype(np.float32))

        if self.verbose:
            print("balanced accuracy: {:.2f}%".format(balanced_accuracy * 100))
            print()

        return balanced_accuracy, unbalanced_accuracy, class_accuracies

    def reset(self):

        # free memory
        if self.session is not None:
            self.__stop_session()

        # reset tf graph
        tf.reset_default_graph()

        # null variables
        self.global_step = None
        self.learning_rate_schedule = None

    def __predict_logits(self, depths, hand_states, actions, labels):
        """
        Generate logit predictions for preprocessed input data.
        :param depths:          Depth images.
        :param hand_states:     Hand states.
        :param actions:         Actions.
        :param labels:          Labels.
        :return:                Logit predictions.
        """

        num_samples = depths.shape[0]
        num_steps = int(np.ceil(num_samples / self.batch_size))

        logits_list = []
        for step_idx in range(num_steps):
            batch_slice = np.index_exp[step_idx * self.batch_size: (step_idx + 1) * self.batch_size]

            logits = self.session.run(self.logits, feed_dict={
                self.depth_pl: depths[batch_slice],
                self.hand_state_pl: hand_states[batch_slice],
                self.actions_pl: actions[batch_slice],
                self.labels_pl: labels[batch_slice],
                self.mask_pl: self.mask,
                self.is_training_pl: False
            })
            logits_list.append(logits)

        logits = np.concatenate(logits_list, axis=0)

        return logits

    def __build_placeholders(self):
        """
        Build Tensorflow placeholders.
        :return:        None.
        """

        self.depth_pl = tf.placeholder(tf.float32, shape=(None, *self.state_shape), name="depth_pl")
        self.hand_state_pl = tf.placeholder(tf.int32, shape=(None,), name="hand_state_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.labels_pl = tf.placeholder(tf.int32, shape=(None,), name="labels_pl")
        self.mask_pl = tf.placeholder(tf.bool, shape=(self.num_classes,), name="mask_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training")

    def __build_network(self):
        """
        Build the ConvNet.
        :return:                None.
        """

        # validate settings
        assert len(self.num_filters_list) == len(self.filter_size_list) == len(self.stride_list)

        # select initializers
        conv_init = None
        fc_init = None
        if self.mrsa_conv_init:
            conv_init = agent_utils.get_mrsa_initializer()
        if self.mrsa_fc_init:
            fc_init = agent_utils.get_mrsa_initializer()

        # encode inputs
        hand_states_one_hot = tf.one_hot(self.hand_state_pl, self.num_hand_states, name="states_to_one_hot")
        actions_one_hot = tf.one_hot(self.actions_pl, self.num_actions, name="actions_to_one_hot")

        # branch 1: hand state and action
        branch_1 = tf.concat([hand_states_one_hot, actions_one_hot], axis=1, name="concat_states_and_actions")

        with tf.variable_scope("embedding"):

            for i, hidden in enumerate(self.hiddens_embed):

                with tf.variable_scope("hidden{:d}".format(i + 1)):

                    if self.batch_norm:
                        activation = None
                        use_bias = False
                    else:
                        activation = tf.nn.relu
                        use_bias = True

                    branch_1 = tf.layers.dense(
                        branch_1, hidden, activation=activation, use_bias=use_bias,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=fc_init
                    )

                    if self.batch_norm and i != len(self.hiddens_embed):
                        branch_1 = tf.layers.batch_normalization(branch_1, training=self.is_training_pl)
                        branch_1 = tf.nn.relu(branch_1)

        # branch 2: depth image
        branch_2 = self.depth_pl

        with tf.variable_scope("convs"):

            for i in range(len(self.num_filters_list)):

                with tf.variable_scope("conv{:d}".format(i + 1)):

                    if self.batch_norm:
                        activation = None
                        use_bias = False
                    else:
                        activation = tf.nn.relu
                        use_bias = True

                    branch_2 = tf.layers.conv2d(
                        branch_2, self.num_filters_list[i], self.filter_size_list[i], self.stride_list[i],
                        padding="SAME", activation=activation, use_bias=use_bias,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=conv_init
                    )

                    if self.batch_norm and i != len(self.num_filters_list):
                        branch_2 = tf.layers.batch_normalization(branch_2, training=self.is_training_pl)
                        branch_2 = tf.nn.relu(branch_2)

        # concatenate branch 1 and 2
        branch_2 = tf.layers.flatten(branch_2, name="flatten_convs")
        x = tf.concat([branch_1, branch_2], axis=1, name="concat_branches")

        if self.batch_norm:
            x = tf.layers.batch_normalization(x, training=self.is_training_pl)
            x = tf.nn.relu(x)

        # final convolutions
        with tf.variable_scope("fcs"):

            for i in range(len(self.hiddens_final)):

                with tf.variable_scope("hidden{:d}".format(i + 1)):

                    if self.batch_norm:
                        activation = None
                        use_bias = False
                    else:
                        activation = tf.nn.relu
                        use_bias = True

                    x = tf.layers.dense(
                        x, self.hiddens_final[i], activation=activation, use_bias=use_bias,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=fc_init
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

        with tf.variable_scope("logits"):

            self.logits = tf.layers.dense(
                x, self.num_classes, kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=fc_init
            )
            self.logits = tf.boolean_mask(self.logits, self.mask_pl, axis=1)
            self.predictions = tf.nn.softmax(self.logits)

            # keep the variables for the output logits, we might need to reset them later
            self.last_layer_variables = []
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                if var.name == "logits/dense/kernel:0" or var.name == "logits/dense/bias:0":
                    self.last_layer_variables.append(var)
            assert len(self.last_layer_variables) == 2
            self.last_layer_init_op = tf.initialize_variables(self.last_layer_variables)

    def __build_training(self):
        """
        Build Tensorflow training operations.
        :return:        None.
        """

        # maybe create global step
        if self.global_step is None:
            self.global_step = tf.train.create_global_step()

        # decide if fixed learning rate or a learning rate schedule should be used
        learning_rate = self.learning_rate
        if self.learning_rate_schedule is not None:
            learning_rate = self.learning_rate_schedule

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_pl, logits=self.logits)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        if len(reg_losses) > 0:
            self.regularization = tf.add_n(reg_losses)
        else:
            self.regularization = tf.constant(0.0, dtype=tf.float32)

        self.loss = self.cross_entropy + self.regularization

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.labels_pl, tf.cast(tf.argmax(self.logits, axis=1), tf.int32)), tf.float32)
        )

        optimizer = tf.train.MomentumOptimizer(learning_rate, self.momentum)

        if self.gradient_clip_norm is not None:
            # clip gradients to some norm
            gradients = optimizer.compute_gradients(self.loss)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, self.gradient_clip_norm), var)
            self.train_step = optimizer.apply_gradients(gradients)
        else:
            self.train_step = optimizer.minimize(
                self.loss, global_step=self.global_step
            )

        if self.batch_norm:
            # running means for batch normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_op = tf.group(*update_ops)
            self.train_step = tf.group(self.train_step, update_op)

    def __start_session(self):
        """
        Start Tensorflow session and initialize all variables.
        :return:
        """

        gpu_options = None
        if self.gpu_memory_fraction is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def __stop_session(self):
        """
        Stop Tensorflow session.
        :return:
        """

        self.session.close()

    def __build_lr_schedule(self):
        """
        Setup a decaying learning rate.
        :return:        None.
        """

        if self.global_step is None:
            self.global_step = tf.train.create_global_step()

        self.learning_rate_schedule = tf.train.exponential_decay(
            self.learning_rate, self.global_step, self.lr_decay_step, self.lr_decay_rate
        )

    def reset_last_layer(self):
        """
        Reset the weights of the classification layer.
        :return:        None.
        """

        self.session.run(self.last_layer_init_op)
