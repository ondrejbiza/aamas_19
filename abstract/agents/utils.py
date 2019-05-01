import os
import numpy as np
import tensorflow as tf
from .. import constants
from . import tf_util_rob as U


def build_get_move_action_descriptors(make_obs_ph, action_shape, action_shape_small, stride):
    """
    Create a function that computes the action descriptors.
    :param make_obs_ph:             Make observation placeholder function.
    :param action_shape:            Shape of the deictic image region.
    :param action_shape_small:      Shape of the deictic image after resizing.
    :param stride:                  Stride with which to sample the deictic images (the default should be the same
                                    as the block size).
    :return:                        A function that computes the action descriptors.
    """

    # get observation
    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    obs = observations_ph.get()

    # get deictic images
    patches = tf.extract_image_patches(
        obs,
        ksizes=[1, action_shape[0], action_shape[1], 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding="SAME")

    # reshape and resize patches
    patches_shape = tf.shape(patches)
    patches_tiled = tf.reshape(
        patches, [patches_shape[0] * patches_shape[1] * patches_shape[2], action_shape[0], action_shape[1], 1]
    )
    patches_tiled_small = tf.image.resize_images(patches_tiled, [action_shape_small[0], action_shape_small[1]])
    patches_tiled_small = tf.reshape(patches_tiled_small, [-1, action_shape_small[0], action_shape_small[1]])

    get_move_action_descriptors = U.function(inputs=[observations_ph], outputs=patches_tiled_small)
    return get_move_action_descriptors


def build_colins_get_move_action_descriptors(make_obs_ph, action_shape, action_shape_small, stride, num_rots=1):

    observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
    obs = observations_ph.get()
    shape = tf.shape(obs)
    deictic_pad = np.int32(2*np.floor(np.array(action_shape)/3))
    obs_zero_padded = tf.image.resize_image_with_crop_or_pad(obs, shape[1]+deictic_pad[0], shape[2]+deictic_pad[1])
    patches = tf.extract_image_patches(
          obs_zero_padded,
          ksizes=[1, action_shape[0], action_shape[1], 1],
          strides=[1, stride, stride, 1],
          rates=[1, 1, 1, 1],
          padding='VALID')
    patches_shape = tf.shape(patches)
    patches_tiled = tf.reshape(patches, [patches_shape[0]*patches_shape[1]*patches_shape[2], action_shape[0],action_shape[1],1])

    rots = np.linspace(0, 16, num_rots, endpoint=False) * np.pi / 16
    if rots.size == 1:
        patches_tiled_rot = patches_tiled
    else:
        patches_tiled_rot = tf.concat([tf.contrib.image.rotate(patches_tiled, rot) for rot in rots], axis=0)

    patches_tiled_small = tf.image.resize_images(patches_tiled_rot, [action_shape_small[0], action_shape_small[1]])
    patches_tiled_small = tf.reshape(patches_tiled_small, [-1,action_shape_small[0],action_shape_small[1]])

    get_move_action_descriptors = U.function(inputs=[observations_ph], outputs=patches_tiled_small)
    return get_move_action_descriptors


def build_get_gamma_deictic(make_action_deic_ph, gamma_func, scope="deep_gamma", gamma_scope="gamma_func", reuse=None,
                            make_abstract_action_ph=None, num_abstract_actions=None, abstract_embedding_size=None):
    """
    Build a model for the gamma function with deictic mappings.
    :param make_action_deic_ph:             Make deictic action placeholder.
    :param gamma_func:                      Neural network function.
    :param scope:                           Tensorflow scope of the neural net.
    :param gamma_scope:                     Scope of the gamma function.
    :param reuse:                           Reuse variables.
    :param make_abstract_action_ph:         Make abstract action placeholder.
    :param num_abstract_actions:            Number of abstract actions.
    :param abstract_embedding_size:         Abstract action embedding size.
    :return:                                Gamma function.
    """

    # make abstract action ph XOR num abstract actions XOR abstract embedding size
    assert (make_abstract_action_ph is None) == (num_abstract_actions is None) == (abstract_embedding_size is None)
    include_abstract_actions = make_abstract_action_ph is not None

    with tf.variable_scope(scope, reuse=reuse):

        actions_ph = U.ensure_tf_input(make_action_deic_ph("actions"))

        if include_abstract_actions:
            abstract_action_ph = U.ensure_tf_input(make_abstract_action_ph("abstract_actions"))
            gamma_values = gamma_func(actions_ph.get(), abstract_action_ph.get(), 1, num_abstract_actions,
                                      abstract_embedding_size, scope=gamma_scope)
            get_gamma = U.function(inputs=[actions_ph, abstract_action_ph], outputs=gamma_values)
        else:
            gamma_values = gamma_func(actions_ph.get(), 1, scope=gamma_scope)
            get_gamma = U.function(inputs=[actions_ph], outputs=gamma_values)

    return get_gamma


def build_gamma_deictic_target_train(make_action_deic_ph, make_target_ph, make_weight_ph, gamma_func, optimizer,
                                     scope="deep_gamma", gamma_scope="gamma_func", grad_norm_clipping=None, reuse=None,
                                     make_abstract_action_ph=None, num_abstract_actions=None,
                                     abstract_embedding_size=None, softmax=False):
    """
    Build an optimizer for the gamma network with deictic mappings.
    :param make_action_deic_ph:             Make a placeholder for deictic action,
    :param make_target_ph:                  Make a placeholder for training target.
    :param make_weight_ph:                  Make a placeholder for training weights.
    :param gamma_func:                      Gamma network.
    :param optimizer:                       Tensorflow optimizer.
    :param scope:                           Tensorflow scope of the network.
    :param gamma_scope:                     Tensorflow scope of the gamma network.
    :param grad_norm_clipping:              Clip gradient norm.
    :param reuse:                           Reuse variables.
    :param make_abstract_action_ph:         Make a placeholder for abstract actions.
    :param num_abstract_actions:            Number of abstract actions.
    :param abstract_embedding_size:         Abstract action embedding size.
    :param softmax:                         Use softmax instead of sigmoid loss function.
    :return:                                Training step function.
    """

    # make abstract action ph XOR num abstract actions XOR abstract embedding size
    assert (make_abstract_action_ph is None) == (num_abstract_actions is None) == (abstract_embedding_size is None)
    include_abstract_actions = make_abstract_action_ph is not None

    with tf.variable_scope(scope, reuse=reuse):

        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_action_deic_ph("action_t_deic"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
        importance_weights_ph = U.ensure_tf_input(make_weight_ph("target"))

        # get variables
        gamma_func_vars = U.scope_vars(U.absolute_scope_name(gamma_scope))

        # gamma values for all actions
        abstract_action_t_input = None
        if include_abstract_actions:
            abstract_action_t_input = U.ensure_tf_input(make_abstract_action_ph("abstract_action_t"))
            gamma_t_raw = gamma_func(obs_t_input.get(), abstract_action_t_input.get(), 1, num_abstract_actions,
                                     abstract_embedding_size, scope=gamma_scope, reuse=True)
        else:
            gamma_t_raw = gamma_func(obs_t_input.get(), 1, scope=gamma_scope, reuse=True)

        target_tiled = tf.reshape(target_input.get(), shape=(-1, 1))

        # calculate error
        if softmax:
            errors = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(target_tiled), logits=gamma_t_raw)
        else:
            errors = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.stop_gradient(target_tiled), logits=gamma_t_raw)
        weighted_errors = importance_weights_ph.get() * errors

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer, weighted_errors, var_list=gamma_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_errors, var_list=gamma_func_vars)

        if include_abstract_actions:
            target_train = U.function(
                inputs=[
                    obs_t_input,
                    abstract_action_t_input,
                    target_input,
                    importance_weights_ph
                ],
                outputs=[errors, obs_t_input.get(), target_input.get()],
                updates=[optimize_expr]
            )
        else:
            target_train = U.function(
                inputs=[
                    obs_t_input,
                    target_input,
                    importance_weights_ph
                ],
                outputs=[errors, obs_t_input.get(), target_input.get()],
                updates=[optimize_expr]
            )

        return target_train


def build_get_f_bar(make_state_ph, f_bar_func, scope="deep_f_bar", f_bar_scope="f_bar_func", reuse=None):
    """
    Build a network that predicts a single proposition.
    :param make_state_ph:       Make state placeholder.
    :param f_bar_func:          Neural network function.
    :param scope:               Tensorflow scope of the neural network.
    :param f_bar_scope:         Tensorflow scope of the f bar function.
    :param reuse:               Reuse variables.
    :return:                    Proposition function.
    """

    with tf.variable_scope(scope, reuse=reuse):

        state_ph = U.ensure_tf_input(make_state_ph("state"))
        f_bar_value = f_bar_func(state_ph.get(), 1, scope=f_bar_scope)
        get_f_bar = U.function(inputs=[state_ph], outputs=f_bar_value)
        return get_f_bar


def build_f_bar_train(make_state_ph, make_target_ph, f_bar_func, optimizer, scope="deep_f_bar",
                      f_bar_scope="f_bar_func", grad_norm_clipping=None, reuse=None):
    """
    Build training for a network that predicts a single proposition.
    :param make_state_ph:           Make state placeholder.
    :param make_target_ph:          Make target placeholder.
    :param f_bar_func:              Neural network function.
    :param optimizer:               Tensorflow optimizer instance.
    :param scope:                   Tensorflow scope of the neural network.
    :param f_bar_scope:             Tensorflow scope of the f bar function.
    :param grad_norm_clipping:      Clip the norm of gradients to this value.
    :param reuse:                   Reuse variables.
    :return:                        Training function.
    """

    with tf.variable_scope(scope, reuse=reuse):

        state_input = U.ensure_tf_input(make_state_ph("state"))
        target_input = U.ensure_tf_input(make_target_ph("target"))

        f_bar_func_vars = U.scope_vars(U.absolute_scope_name(f_bar_scope))

        f_bar_raw = f_bar_func(state_input.get(), 1, scope=f_bar_scope, reuse=True)
        target_tiled = tf.reshape(target_input.get(), shape=(-1, 1))

        errors = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.stop_gradient(target_tiled), logits=f_bar_raw)
        is_correct = tf.equal(target_tiled, tf.cast(tf.greater(f_bar_raw, tf.constant(0.0, dtype=tf.float32)),
                                                    dtype=tf.float32))

        if grad_norm_clipping:
            optimize_expr = U.minimize_and_clip(optimizer, errors, var_list=f_bar_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(errors, var_list=f_bar_func_vars)

        f_bar_train = U.function(
            inputs=[state_input, target_input],
            outputs=[errors, is_correct],
            updates=[optimize_expr]
        )

        return f_bar_train


def build_getq(input_ph, q_func, scope="deepq", qscope="q_func", num_actions=1, reuse=None):
    """
    Build q-network.
    :param input_ph:                Get input placeholder.
    :param q_func:                  Build q-network.
    :param scope:                   Scope of the q-network.
    :param qscope:                  Scope of the q-function.
    :param reuse:                   Reuse Tensorflow variables.
    :return:                        Get q function.
    """

    with tf.variable_scope(scope, reuse=reuse):

        actions_ph = U.ensure_tf_input(input_ph("actions"))
        q_values = q_func(actions_ph.get(), num_actions, scope=qscope)
        getq = U.function(inputs=[actions_ph], outputs=q_values)

        return getq


def build_getq_branch(depth_ph, hand_ph, num_actions, q_func, scope="deepq", qscope="q_func", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        depth_ph = U.ensure_tf_input(depth_ph("depths"))
        hand_ph = U.ensure_tf_input(hand_ph("hand_states"))

        q_values = q_func(depth_ph.get(), hand_ph.get(), num_actions, scope=qscope)
        getq = U.function(inputs=[depth_ph, hand_ph], outputs=q_values)

        return getq



def build_q_deictic_target_train(make_action_deic_ph, make_target_ph, make_weight_ph, q_func, optimizer, scope="deepq",
                                 qscope="q_func", grad_norm_clipping=None, reuse=None):
    """
    Build training for q-network.
    :param make_action_deic_ph:     Get deictic action image placeholder.
    :param make_target_ph:          Get target placeholder.
    :param make_weight_ph:          Get target weights placeholder.
    :param q_func:                  Build q-network.
    :param optimizer:               Get Tensorflow optimizer.
    :param scope:                   Scope of the q-network.
    :param qscope:                  Scope of the q-function.
    :param grad_norm_clipping:      Clip gradient norm.
    :param reuse:                   Reuse Tensorflow variables.
    :return:
    """

    with tf.variable_scope(scope, reuse=reuse):

        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_action_deic_ph("action_t_deic"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
        importance_weights_ph = U.ensure_tf_input(make_weight_ph("target"))

        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))

        # q values for all actions
        q_t_raw = q_func(obs_t_input.get(), 1, scope=qscope, reuse=True)
        target_tiled = tf.reshape(target_input.get(), shape=(-1, 1))

        # calculate error
        td_error = q_t_raw - tf.stop_gradient(target_tiled)
        errors = importance_weights_ph.get() * U.huber_loss(td_error)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer, errors, var_list=q_func_vars, clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)

        target_train = U.function(
            inputs=[
                obs_t_input,
                target_input,
                importance_weights_ph
            ],
            outputs=[td_error, obs_t_input.get(), target_input.get()],
            updates=[optimize_expr]
        )

        return target_train


def build_q_target_train(make_states_ph, make_actions_ph, make_target_ph, make_weight_ph, q_func, optimizer,
                         num_actions, scope="deepq", qscope="q_func", grad_norm_clipping=None, reuse=None):
    """
    Build training for q-network.
    :param make_states_ph:          Get states placeholder.
    :param make_actions_ph:         Get actions placeholder.
    :param make_target_ph:          Get target placeholder.
    :param make_weight_ph:          Get target weights placeholder.
    :param q_func:                  Build q-network.
    :param optimizer:               Get Tensorflow optimizer.
    :param num_actions:             Number of actions to predict.
    :param scope:                   Scope of the q-network.
    :param qscope:                  Scope of the q-function.
    :param grad_norm_clipping:      Clip gradient norm.
    :param reuse:                   Reuse Tensorflow variables.
    :return:
    """

    with tf.variable_scope(scope, reuse=reuse):

        # set up placeholders
        states_input = U.ensure_tf_input(make_states_ph("states"))
        actions_input = U.ensure_tf_input(make_actions_ph("actions"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
        importance_weights_ph = U.ensure_tf_input(make_weight_ph("weights"))

        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))

        # q values for all actions
        q_t_raw = q_func(states_input.get(), num_actions, scope=qscope, reuse=True)
        mask = tf.one_hot(actions_input.get(), num_actions, dtype=tf.float32)
        q_t_raw = tf.reduce_sum(q_t_raw * mask, axis=1)

        # calculate error
        td_error = q_t_raw - tf.stop_gradient(target_input.get())
        errors = importance_weights_ph.get() * U.huber_loss(td_error)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer, errors, var_list=q_func_vars, clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)

        target_train = U.function(
            inputs=[
                states_input,
                actions_input,
                target_input,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )

        return target_train


def build_q_branch_target_train(make_depths_ph, make_hand_ph, make_actions_ph, make_target_ph, make_weight_ph, q_func,
                                optimizer, num_actions, scope="deepq", qscope="q_func", grad_norm_clipping=None,
                                reuse=None):

    with tf.variable_scope(scope, reuse=reuse):

        # set up placeholders
        make_depths_ph = U.ensure_tf_input(make_depths_ph("depths"))
        make_hand_ph = U.ensure_tf_input(make_hand_ph("hand_states"))
        actions_input = U.ensure_tf_input(make_actions_ph("actions"))
        target_input = U.ensure_tf_input(make_target_ph("target"))
        importance_weights_ph = U.ensure_tf_input(make_weight_ph("weights"))

        # get variables
        q_func_vars = U.scope_vars(U.absolute_scope_name(qscope))

        # q values for all actions
        q_t_raw = q_func(make_depths_ph.get(), make_hand_ph.get(), num_actions, scope=qscope, reuse=True)
        mask = tf.one_hot(actions_input.get(), num_actions, dtype=tf.float32)
        q_t_raw = tf.reduce_sum(q_t_raw * mask, axis=1)

        # calculate error
        td_error = q_t_raw - tf.stop_gradient(target_input.get())
        errors = importance_weights_ph.get() * U.huber_loss(td_error)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer, errors, var_list=q_func_vars, clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)

        target_train = U.function(
            inputs=[
                make_depths_ph,
                make_hand_ph,
                actions_input,
                target_input,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )

        return target_train


def embed(index_t, num_items, embedding_size):
    """
    Embed discrete input.
    :param index_t:             Index input.
    :param num_items:           Number of items.
    :param embedding_size:      Embedding size.
    :return:                    Input embedding.
    """

    embeddings_var = tf.get_variable("embeddings", [num_items, embedding_size], trainable=True)
    embedded_t = tf.nn.embedding_lookup(embeddings_var, index_t)

    return embedded_t


def get_optimizer(optimizer, learning_rate):
    """
    Get a Tensorflow optimizer.
    :param optimizer:           Optimizer key from constants.py.
    :param learning_rate:       Learning rate (can be a float or a Tensor).
    :return:                    An instance of a Tensorflow optimizer.
    """

    if optimizer == constants.OPT_ADAM:
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer == constants.OPT_MOMENTUM:
        return tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate)


def get_mrsa_initializer():
    """
    Get MRSA (Microsoft Research Asia) ConvNet weights initializer.
    :return:      The initializer object.
    """

    return tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode="FAN_IN", uniform=False)


def get_weight_regularizer(weight_decay):
    """
    Get L2 weight regularizer.
    :param weight_decay:    Weight decay.
    :return:                Regularizer object.
    """
    return tf.contrib.layers.l2_regularizer(weight_decay)


def process_states(states):
    """
    Separate list of states into lists of depths and hand states.
    :param states:      List of states.
    :return:            List of depths and list of hand states; each pair is from the same state.
    """

    depths = []
    hand_states = []

    for state in states:
        depths.append(state[0])
        hand_states.append(state[1])

    depths = np.array(depths, dtype=np.float32)
    hand_states = np.array(hand_states, dtype=np.int32)

    return depths, hand_states


def gaussian_kernel(size, mean, std):
    """
    https://stackoverflow.com/questions/52012657/how-to-make-a-2d-gaussian-filter-in-tensorflow
    """

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def summarize(tensor, name):

    s1 = tf.summary.histogram(name + "_hist", tensor)
    s2 = tf.summary.scalar(name + "_min", tf.reduce_min(tensor))
    s3 = tf.summary.scalar(name + "_max", tf.reduce_max(tensor))

    return s1, s2, s3


def new_run_dir(base, dir_name="run"):

    idx = 1
    while True:

        path = os.path.join(base, "{}{}".format(dir_name, idx))

        if not os.path.isdir(path):
            os.makedirs(path)
            return path
        else:
            idx += 1
