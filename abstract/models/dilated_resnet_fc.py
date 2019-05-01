import time
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.client import timeline
from ..agents import utils as agent_utils
from . import utils as balancing_utils
from ..tf_calibrate import calibration as calibration


class DilatedResnetFC:

    NUM_HAND_STATES = 2
    MOMENTUM = 0.9

    def __init__(self, state_shape, output_shape, num_filters_list, filter_size_list, stride_list, dilation,
                 block_num_filters_list, block_strides_list, block_dilation,
                 num_post_filters, post_filter_size_list, post_stride_list, post_dilation,
                 weight_decay=0.0001, learning_rate=0.01, learning_rate_steps=None, learning_rate_values=None,
                 gpu_memory_fraction=0.2, preprocess=None, validation_fraction=0.2, balance=None, verbose=False,
                 set_best_params=True, batch_size=32, max_training_steps=5000, validation_frequency=100, g_mean=False,
                 early_stop_no_improvement_steps=1000, final_filter_size=1, batch_norm=False, upsample_before=False,
                 upsample_after=False, deconv_before=False, deconv_after=False, deconv_filter_size=5,
                 deconv_before_num_filters=32, profile_path=None, valid_batch_size=1024, calibrate=False,
                 gauss_smooth=False, gauss_smooth_std=1.0, gauss_smooth_size=5, gauss_smooth_logits=False):

        assert np.sum([upsample_before, upsample_after, deconv_before, deconv_after]) <= 1

        self.state_shape = state_shape
        self.output_shape = output_shape
        self.num_filters_list = num_filters_list
        self.filter_size_list = filter_size_list
        self.stride_list = stride_list
        self.dilation = dilation
        self.block_num_filters_list = block_num_filters_list
        self.block_strides_list = block_strides_list
        self.block_dilation = block_dilation
        self.num_post_filters = num_post_filters
        self.post_filter_size_list = post_filter_size_list
        self.post_stride_list = post_stride_list
        self.post_dilation = post_dilation
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.learning_rate_steps = learning_rate_steps
        self.learning_rate_values = learning_rate_values
        self.gpu_memory_fraction = gpu_memory_fraction
        self.preprocess = preprocess
        self.validation_fraction = validation_fraction
        self.balance = balance
        self.verbose = verbose
        self.set_best_params = set_best_params
        self.batch_size = batch_size
        self.max_training_steps = max_training_steps
        self.validation_frequency = validation_frequency
        self.g_mean = g_mean
        self.early_stop_no_improvement_steps = early_stop_no_improvement_steps
        self.final_filter_size = final_filter_size
        self.batch_norm = batch_norm
        self.upsample_before = upsample_before
        self.upsample_after = upsample_after
        self.deconv_before = deconv_before
        self.deconv_after = deconv_after
        self.deconv_filter_size = deconv_filter_size
        self.deconv_before_num_filters = deconv_before_num_filters
        self.profile_path = profile_path
        self.valid_batch_size = valid_batch_size
        self.calibrate = calibrate
        self.gauss_smooth = gauss_smooth
        self.gauss_smooth_std = gauss_smooth_std
        self.gauss_smooth_size = gauss_smooth_size
        self.gauss_smooth_logits = gauss_smooth_logits

        self.tf_graph = tf.Graph()

        self.global_step = None
        self.session = None
        self.total_num_training_steps = 0

        self.options = None
        self.run_metadata = None
        if self.profile_path is not None:
            # https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

        self.logits = None
        self.logits_flat = None
        self.predictions_flat = None

        self.temperature = None
        self.scaled_logits_flat = None
        self.scaled_predictions_flat = None

    def predict(self, states, actions, apply_softmax=True, disable_preprocess=False):
        """
        Predict labels for given data.
        :param states:          States
        :param actions:         Actions.
        :param apply_softmax:   Apply softmax to get a valid probability distribution.
        :return:                Predictions.
        """

        # preprocess data
        if self.preprocess is not None and not disable_preprocess:
            depths, hand_states = self.preprocess(states)
        else:
            depths = states[0]
            hand_states = states[1]

        if apply_softmax:
            if self.calibrate and self.scaled_predictions_flat is not None:
                to_run = self.scaled_predictions_flat
            else:
                to_run = self.predictions_flat
        else:
            if self.calibrate and self.scaled_logits_flat is not None:
                to_run = self.scaled_logits_flat
            else:
                to_run = self.logits_flat

        num_samples = depths.shape[0]
        num_steps = int(np.ceil(num_samples / self.valid_batch_size))

        predictions_list = []

        with self.tf_graph.as_default():
            for step_idx in range(num_steps):
                tmp_predictions = self.session.run(to_run, feed_dict={
                    self.depth_pl: depths[step_idx * self.valid_batch_size:(step_idx + 1) * self.valid_batch_size],
                    self.hand_states_pl: hand_states[step_idx * self.valid_batch_size:
                                                     (step_idx + 1) * self.valid_batch_size],
                    self.is_training_pl: False
                })
                predictions_list.append(tmp_predictions)

        predictions = np.concatenate(predictions_list, axis=0)

        selected_preds = []
        for prediction, action in zip(predictions, actions):
            selected_preds.append(prediction[action])
        selected_preds = np.array(selected_preds, dtype=np.float32)

        return selected_preds

    def predict_all_actions(self, states, apply_softmax=True, flat=True):

        # preprocess data
        if self.preprocess is not None:
            depths, hand_states = self.preprocess(states)
        else:
            depths = states[0]
            hand_states = states[1]

        if apply_softmax:
            if flat:
                if self.calibrate and self.scaled_predictions_flat is not None:
                    to_run = self.scaled_predictions_flat
                else:
                    to_run = self.predictions_flat
            else:
                to_run = self.predictions
        else:
            if flat:
                if self.calibrate and self.scaled_logits_flat is not None:
                    to_run = self.scaled_logits_flat
                else:
                    to_run = self.logits_flat
            else:
                to_run = self.logits

        # predict
        num_samples = depths.shape[0]
        num_steps = int(np.ceil(num_samples / self.valid_batch_size))

        predictions_list = []

        with self.tf_graph.as_default():
            for step_idx in range(num_steps):
                tmp_predictions = self.session.run(to_run, feed_dict={
                    self.depth_pl: depths[step_idx * self.valid_batch_size:(step_idx + 1) * self.valid_batch_size],
                    self.hand_states_pl: hand_states[step_idx * self.valid_batch_size:
                                                     (step_idx + 1) * self.valid_batch_size],
                    self.is_training_pl: False
                })
                predictions_list.append(tmp_predictions)

        predictions = np.concatenate(predictions_list, axis=0)

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

            return self.fit(states, actions, labels, None, None, None)
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

            return self.fit(train_states, train_actions, train_labels, valid_states, valid_actions, valid_labels)

    def fit(self, train_states, train_actions, train_labels, valid_states, valid_actions, valid_labels):
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
        self.num_classes = np.max(train_labels) + 1

        if self.verbose:
            for i in range(self.num_classes):
                print("class {:d}: {:d} train, {:d} validation samples".format(
                    i, np.sum(train_labels == i), np.sum(valid_labels == i))
                )

        # check if the neural network was trained before
        if self.session is None:
            first_run = True
        else:
            first_run = False

        # reset
        self.reset()

        with self.tf_graph.as_default():
            # build placeholders
            self.build_placeholders()

            # build network
            self.build_network()
            self.build_training()

        # start session
        self.start_session()

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

            with self.tf_graph.as_default():
                for idx, variable in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
                    self.session.run(variable.assign(best_parameters[idx]))

        # maybe calibrate confidences
        if self.calibrate:

            with self.tf_graph.as_default():

                logits = self.predict([valid_depths, valid_hand_states], valid_actions, apply_softmax=False,
                                      disable_preprocess=True)

                self.temperature = calibration.temperature_scale(logits, self.session, valid_labels)

                self.scaled_logits_flat = self.logits_flat / self.temperature
                self.scaled_predictions_flat = tf.nn.softmax(self.scaled_logits_flat, axis=-1)

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
        mask = self.batch_mask(train_actions)

        best_balanced_accuracy = None
        best_unbalanced_accuracy = None
        best_class_accuracies = None
        best_step = None
        best_parameters = None

        losses = []
        epoch_start_time = time.time()

        for step_idx in range(self.max_training_steps):

            epoch_step_idx = step_idx % num_steps_per_epoch

            if epoch_step_idx == 0:
                # new epoch, reshuffle dataset
                train_depths, train_hand_states, train_actions, train_labels, mask = shuffle(
                    train_depths, train_hand_states, train_actions, train_labels, mask
                )
            # build feed dictionary
            batch_slice = np.index_exp[epoch_step_idx * self.batch_size:(epoch_step_idx + 1) * self.batch_size]

            feed_dict = {
                self.depth_pl: train_depths[batch_slice],
                self.hand_states_pl: train_hand_states[batch_slice],
                self.labels_pl: train_labels[batch_slice],
                self.mask_pl: mask[batch_slice],
                self.is_training_pl: True
            }

            # take training step
            with self.tf_graph.as_default():
                _, loss = self.session.run([self.train_step, self.loss], feed_dict=feed_dict, options=self.options,
                                           run_metadata=self.run_metadata)
            losses.append(loss)

            if self.profile_path is not None and step_idx == 1:

                fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(self.profile_path, "w") as f:
                    f.write(chrome_trace)

            if validate and step_idx > 0 and step_idx % self.validation_frequency == 0:

                epoch_duration = time.time() - epoch_start_time
                valid_start_time = time.time()

                # perform validation
                if self.verbose:
                    print()
                    print("mean training loss: {:.8f}".format(np.mean(losses)))
                    print("validation, step {:d}".format(step_idx))

                losses = []

                balanced_accuracy, unbalanced_accuracy, class_accuracies = self.validate(
                    valid_depths, valid_hand_states, valid_actions, valid_labels
                )

                if best_balanced_accuracy is None or balanced_accuracy > best_balanced_accuracy:

                    best_balanced_accuracy = balanced_accuracy
                    best_unbalanced_accuracy = unbalanced_accuracy
                    best_class_accuracies = class_accuracies
                    best_step = step_idx

                    if self.set_best_params:
                        with self.tf_graph.as_default():
                            best_parameters = self.session.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

                valid_duration = time.time() - valid_start_time
                if self.verbose:
                    print("epoch duration: {:.2f} seconds".format(epoch_duration))
                    print("validation duration: {:.2f} seconds".format(valid_duration))
                epoch_start_time = time.time()

            elif not validate and step_idx > 0 and step_idx % 100 == 0:

                # print only the training loss
                if self.verbose:
                    print("mean training loss: {:.8f}".format(np.mean(losses)))
                    print("validation, step {:d}".format(step_idx))

                losses = []

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

        logits = self.predict([valid_depths, valid_hand_states], valid_actions, apply_softmax=False,
                              disable_preprocess=True)

        class_accuracies = []
        for class_idx in range(self.num_classes):
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

        return balanced_accuracy, unbalanced_accuracy, class_accuracies

    def reset(self):
        """
        Reset the neural network.
        :return:        None.
        """

        # free memory
        if self.session is not None:
            self.stop_session()
            self.session = None

        self.global_step = None

        self.tf_graph = tf.Graph()

        self.temperature = None
        self.scaled_logits_flat = None
        self.scaled_predictions_flat = None

    def batch_mask(self, actions):
        """
        Create a mask for the predictions.
        :param actions:     List of action indices.
        :return:
        """

        assert np.max(actions) < (self.output_shape[0] * self.output_shape[1])

        num_samples = len(actions)

        mask = np.zeros((num_samples, self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)

        for idx, action in zip(range(len(actions)), actions):

            row = action // self.output_shape[0]
            column = action % self.output_shape[1]

            mask[idx, row, column, :] = 1

        return mask

    def build_placeholders(self):
        """
        Build Tensorflow placeholders.
        :return:        None.
        """

        self.depth_pl = tf.placeholder(tf.float32, shape=(None, *self.state_shape), name="depth_pl")
        self.hand_states_pl = tf.placeholder(tf.int32, shape=(None,), name="hand_states_pl")
        self.labels_pl = tf.placeholder(tf.int32, shape=(None,), name="labels_pl")
        self.mask_pl = tf.placeholder(
            tf.float32, shape=(None, self.output_shape[0], self.output_shape[1], self.num_classes), name="mask_pl"
        )
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

    def build_network(self):
        """
        Build the network. Build the placeholders first.
        :return:        None.
        """

        x = self.depth_pl

        with tf.variable_scope("pre-convs"):

            for i in range(len(self.num_filters_list)):

                with tf.variable_scope("conv{:d}".format(i + 1)):

                    if self.dilation is not None:
                        tmp_dilation = (self.dilation[i], self.dilation[i])
                    else:
                        tmp_dilation = (1, 1)

                    if self.batch_norm:
                        x = tf.layers.conv2d(
                            x, self.num_filters_list[i], self.filter_size_list[i], self.stride_list[i],
                            padding="SAME", activation=None, use_bias=False,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer(),
                            dilation_rate=tmp_dilation
                        )
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)
                    else:
                        x = tf.layers.conv2d(
                            x, self.num_filters_list[i], self.filter_size_list[i], self.stride_list[i],
                            padding="SAME", activation=tf.nn.relu,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer(),
                            dilation_rate=tmp_dilation
                        )

        with tf.variable_scope("blocks"):

            for i in range(len(self.block_num_filters_list)):

                with tf.variable_scope("block{:d}".format(i + 1)):

                    x = self.residual_block(
                        x, self.block_num_filters_list[i], dilation=self.block_dilation[i],
                        stride=self.block_strides_list[i]
                    )

        with tf.variable_scope("post-convs"):

            for i in range(len(self.num_post_filters)):

                with tf.variable_scope("conv{:d}".format(i + 1)):

                    if self.post_dilation is not None:
                        tmp_dilation = (self.post_dilation[i], self.post_dilation[i])
                    else:
                        tmp_dilation = (1, 1)

                    if self.batch_norm:
                        x = tf.layers.conv2d(
                            x, self.num_post_filters[i], self.post_filter_size_list[i], self.post_stride_list[i],
                            padding="SAME", activation=None, use_bias=False,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer(),
                            dilation_rate=tmp_dilation
                        )
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)
                    else:
                        x = tf.layers.conv2d(
                            x, self.num_post_filters[i], self.post_filter_size_list[i], self.post_stride_list[i],
                            padding="SAME", activation=tf.nn.relu,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer(),
                            dilation_rate=tmp_dilation
                        )

        # upsample the output or throw an error
        if x.shape[1] != self.output_shape[0] or x.shape[2] != self.output_shape[1]:

            if self.upsample_before:

                with tf.variable_scope("upsample_before"):
                    x = tf.image.resize_bilinear(x, (self.output_shape[0], self.output_shape[1]), name="upsample")

            elif self.deconv_before:

                with tf.variable_scope("deconv_before"):
                    ratio = self.output_shape[0] // x.shape[1].value
                    x = tf.layers.conv2d_transpose(
                        x, self.deconv_before_num_filters, self.deconv_filter_size,
                        strides=(ratio, ratio), padding="SAME", activation=tf.nn.relu,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer()
                    )

        with tf.variable_scope("logits"):

            # predict for both hand empty and hand full
            self.logits = tf.layers.conv2d(
                x, self.num_classes * 2, self.final_filter_size, 1, padding="SAME", activation=None,
                kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=agent_utils.get_mrsa_initializer()
            )

            # maybe upsample logits
            if self.logits.shape[1] != self.output_shape[0] or self.logits.shape[2] != self.output_shape[1]:

                if self.upsample_after:

                    with tf.variable_scope("upsample_after"):
                        self.logits = tf.image.resize_bilinear(
                            self.logits, (self.output_shape[0], self.output_shape[1]), name="upsample_logits"
                        )

                elif self.deconv_after:

                    with tf.variable_scope("deconv_after"):

                        ratio = self.output_shape[0] // x.shape[1].value

                        with tf.variable_scope("hand_empty"):
                            logits_1 = tf.layers.conv2d_transpose(
                                self.logits[:, :, :, :self.num_classes], self.num_classes, self.deconv_filter_size,
                                strides=(ratio, ratio), padding="SAME",
                                kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                                kernel_initializer=agent_utils.get_mrsa_initializer()
                            )

                        with tf.variable_scope("hand_full"):
                            logits_2 = tf.layers.conv2d_transpose(
                                self.logits[:, :, :, self.num_classes:], self.num_classes,
                                self.deconv_filter_size, strides=(ratio, ratio), padding="SAME",
                                kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                                kernel_initializer=agent_utils.get_mrsa_initializer()
                            )

                        self.logits = tf.concat([logits_1, logits_2], axis=-1)

            assert self.logits.shape[-1].value == self.num_classes * 2

            mask = tf.cast(self.hand_states_pl, tf.bool)
            self.logits = tf.where(
                mask, x=self.logits[:, :, :, :self.num_classes], y=self.logits[:, :, :, self.num_classes:]
            )

            assert self.logits.shape[1].value == self.output_shape[0]
            assert self.logits.shape[2].value == self.output_shape[1]

            # maybe smooth predictions
            if self.gauss_smooth:

                with tf.variable_scope("smoothing"):
                    kernel = agent_utils.gaussian_kernel(self.gauss_smooth_size, 0.0, self.gauss_smooth_std)
                    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
                    kernel = tf.tile(kernel, [1, 1, self.num_classes, 1])

                if self.gauss_smooth_logits:
                    # predictions with smoothed logits
                    with tf.variable_scope("smoothing"):
                        logits = tf.nn.depthwise_conv2d(self.logits, kernel, strides=[1, 1, 1, 1], padding="SAME")
                    self.predictions = tf.nn.softmax(logits, axis=-1)
                else:
                    # predictions with smoothing
                    self.predictions = tf.nn.softmax(self.logits, axis=-1)
                    with tf.variable_scope("smoothing"):
                        self.predictions = tf.nn.conv2d(self.predictions, kernel, strides=[1, 1, 1, 1], padding="SAME")

            else:
                # predictions without smoothing
                self.predictions = tf.nn.softmax(self.logits, axis=-1)

            self.logits_flat = tf.reshape(
                self.logits, shape=(-1, self.output_shape[0] * self.output_shape[1], self.num_classes)
            )
            self.predictions_flat = tf.reshape(
                self.predictions, shape=(-1, self.output_shape[0] * self.output_shape[1], self.num_classes)
            )

    def build_training(self):
        """
        Build the training. Build the placeholders and the network first.
        :return:            None.
        """

        # maybe create global step
        if self.global_step is None:
            self.global_step = tf.train.create_global_step()

        # mask and reshape logits for training
        logits_mask = self.logits * self.mask_pl
        logits_mask = tf.reduce_sum(logits_mask, axis=[1, 2])

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_pl, logits=logits_mask)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy)

        if self.weight_decay > 0.0 and len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            self.regularization = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = self.cross_entropy + self.regularization
        else:
            self.loss = self.cross_entropy

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.labels_pl, tf.cast(tf.argmax(logits_mask, axis=1), tf.int32)), tf.float32)
        )

        self.learning_rate_steps_to_tensor()

        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.MOMENTUM)

        self.train_step = optimizer.minimize(
            self.loss, global_step=self.global_step
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.train_step = tf.group(self.train_step, self.update_op)

    def start_session(self):
        """
        Start Tensorflow session and initialize all variables.
        :return:
        """

        with self.tf_graph.as_default():
            gpu_options = None
            if self.gpu_memory_fraction is not None:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)

            tf_config = tf.ConfigProto(gpu_options=gpu_options)

            self.session = tf.Session(config=tf_config)
            self.session.run(tf.global_variables_initializer())

    def stop_session(self):
        """
        Stop Tensorflow session.
        :return:
        """

        with self.tf_graph.as_default():
            self.session.close()

    def visualize_predictions(self, states, hand_states, labels, save_path, samples_per_class,
                              num_columns=5):

        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_pdf
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import vis_utils

        num_rows = int(np.ceil(float(samples_per_class) / num_columns))
        max_height = np.max(states)
        addition = max_height / 3

        figures = []
        all_classes = np.unique(labels)

        for cls in sorted(all_classes):

            fig, axes = plt.subplots(nrows=num_rows * 2, ncols=num_columns)
            figures.append(fig)

            if len(axes.shape) == 1:
                axes = np.expand_dims(axes, axis=0)

            fig.set_figheight(24)
            fig.set_figwidth(6)
            fig.suptitle("class {:d}".format(cls))

            indices = np.argwhere(labels == cls)[:, 0]

            samples = np.random.choice(indices, size=samples_per_class, replace=False)

            cls_states = states[samples]
            cls_hand_states = hand_states[samples]

            predictions = self.predict_all_actions([cls_states, cls_hand_states], apply_softmax=True, flat=True)

            idx = 0
            for state, hand_state, prediction in zip(cls_states, cls_hand_states, predictions):

                classes = np.argmax(prediction, axis=1)

                values = classes + prediction[list(range(len(classes))), classes]
                values = np.reshape(values, (state.shape[0], state.shape[1]))

                image = vis_utils.multiplex_hand_state([state, hand_state], max_height + addition)

                axis = axes[(idx // num_columns) * 2, idx % num_columns]
                axis.imshow(image[:, :, 0], vmin=0, vmax=max_height + addition)
                axis.axis("off")

                axis = axes[(idx // num_columns) * 2 + 1, idx % num_columns]
                im = axis.imshow(values, vmin=0, vmax=np.max(all_classes) + 1)

                divider = make_axes_locatable(axis)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                cbar.ax.tick_params(labelsize=5)

                axis.axis("off")

                idx += 1

        pdf = matplotlib.backends.backend_pdf.PdfPages(save_path)

        for fig in figures:
            pdf.savefig(fig)

        pdf.close()

    def residual_block(self, x, filters_out, identity_shortcut=True, dilation=1, stride=1):

        filters_in = x.shape[-1].value

        residual = x

        with tf.variable_scope("A"):
            x = tf.layers.conv2d(x, filters_out, kernel_size=3, strides=(stride, stride), use_bias=False,
                                 kernel_initializer=agent_utils.get_mrsa_initializer(), padding="same",
                                 kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                                 dilation_rate=(dilation, dilation))
            x = tf.layers.batch_normalization(x, training=self.is_training_pl)
            x = tf.nn.relu(x)

        with tf.variable_scope("B"):
            x = tf.layers.conv2d(x, filters_out, kernel_size=3, strides=(1, 1), use_bias=False,
                                 kernel_initializer=agent_utils.get_mrsa_initializer(), padding="same",
                                 kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                                 dilation_rate=(dilation, dilation))
            x = tf.layers.batch_normalization(x, training=self.is_training_pl)

        with tf.variable_scope("residual"):
            if filters_out != filters_in or stride != 1:
                if identity_shortcut:
                    if stride != 1:
                        residual = tf.layers.max_pooling2d(residual, (1, 1), (stride, stride), padding="same")
                    if filters_out != filters_in:
                        residual = tf.pad(residual, [[0, 0], [0, 0], [0, 0], [0, filters_out - filters_in]])
                else:
                    residual = tf.layers.conv2d(
                        residual, filters_out, kernel_size=1, strides=(stride, stride), use_bias=False,
                        kernel_initializer=agent_utils.get_mrsa_initializer(), padding="same",
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        dilation_rate=(dilation, dilation)
                    )
                    residual = tf.layers.batch_normalization(residual, training=self.is_training_pl)

        merge = tf.add(x, residual, name="residual_add")
        activate = tf.nn.relu(merge, name="final_relu")

        return activate

    def learning_rate_steps_to_tensor(self):

        if self.learning_rate_steps is not None and self.learning_rate_values is not None:
            self.learning_rate = tf.train.piecewise_constant(
                self.global_step, self.learning_rate_steps, self.learning_rate_values
            )
