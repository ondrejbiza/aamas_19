import tensorflow as tf
import tensorflow.contrib.layers as layers


def _mlp(hiddens, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


def mlp(hiddens=[]):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, *args, **kwargs)

def embed(index_t, num_items, embedding_size):

    embeddings_var = tf.get_variable("embeddings", [num_items, embedding_size], trainable=True)
    embedded_t = tf.nn.embedding_lookup(embeddings_var, index_t)

    return embedded_t

def multiplex(vector_t, size):
    """
    Go from batch_size x num_filters to batch_size x size x size x num_filters by repeating the vector
    size ^ 2 times.
    :param vector_t:    Batch of input vectors.
    :param size:        Height and width.
    :return:            Tensor of rank 4 that can be added to the output of the convolution.
    """

    vector_t = tf.expand_dims(vector_t, axis=1)
    vector_t = tf.expand_dims(vector_t, axis=1)

    vector_t = tf.tile(vector_t, (1, size, size, 1))

    return vector_t

def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores

def _cnn_to_mlp_state_direct(convs, hiddens, dueling, action_input, state_input, num_states, num_actions, scope,
                             reuse=False):

    state_input = tf.one_hot(state_input, num_states, dtype=tf.float32)

    with tf.variable_scope(scope, reuse=reuse):
        out = action_input
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

        out = layers.flatten(out)
        out = tf.concat((out, state_input), axis=1)

        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores


def _cnn_to_mlp_state_embed(convs, hiddens, dueling, action_input, state_input, num_states, state_embedding_size,
                            num_actions, scope, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):
        out = action_input
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

        with tf.variable_scope("embedding"):

            embedded_state_t = embed(state_input, num_states, state_embedding_size)
            embedded_state_t = tf.nn.relu(embedded_state_t)

        out = layers.flatten(out)
        out = tf.concat((out, embedded_state_t), axis=1)

        with tf.variable_scope("action_value"):
            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            return state_score + action_scores_centered
        else:
            return action_scores


def _branch_cnn_to_mlp(convs, hand_hiddens, final_hiddens, depth_input, hand_input, num_actions, scope, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):

        depth_x = depth_input
        hand_x = tf.one_hot(hand_input, depth=2, dtype=tf.float32)

        with tf.variable_scope("convnet"):

            for num_outputs, kernel_size, stride in convs:
                depth_x = layers.convolution2d(
                    depth_x, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, activation_fn=tf.nn.relu
                )

        with tf.variable_scope("action_mlp"):

            for num_neurons in hand_hiddens:
                hand_x = layers.fully_connected(
                    hand_x, num_neurons, activation_fn=tf.nn.relu
                )

        depth_x = layers.flatten(depth_x)
        final_x = tf.concat([depth_x, hand_x], axis=1)

        with tf.variable_scope("final_mlp"):

            for num_neurons in final_hiddens:
                final_x = layers.fully_connected(
                    final_x, num_neurons, activation_fn=tf.nn.relu
                )

            action_scores = layers.fully_connected(final_x, num_outputs=num_actions, activation_fn=None)

        return action_scores


def _cnn_to_mlp_symbolic(convs, hiddens, deictic_input, abstract_input, num_actions, num_abstract_actions,
                         abstract_embedding_size, scope, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):

        out = deictic_input
        with tf.variable_scope("convnet"):

            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

        out = layers.flatten(out)

        with tf.variable_scope("embedding"):

            embedded_abstract_action_t = embed(abstract_input, num_abstract_actions, abstract_embedding_size)
            embedded_abstract_action_t = tf.nn.relu(embedded_abstract_action_t)

        out = tf.concat((out, embedded_abstract_action_t), axis=1)

        with tf.variable_scope("action_value"):

            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        return action_scores


def _cnn_to_mlp_symbolic_multiplex(convs, hiddens, deictic_input, abstract_input, num_actions, num_abstract_actions,
                                   abstract_embedding_size, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):

        out = deictic_input
        with tf.variable_scope("convnet"):

            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

        with tf.variable_scope("embedding"):

            embedded_abstract_action_t = embed(abstract_input, num_abstract_actions, abstract_embedding_size)
            embedded_abstract_action_t = tf.nn.relu(embedded_abstract_action_t)

            multiplex_t = multiplex(embedded_abstract_action_t, tf.shape(out)[1])

        out += multiplex_t
        out = layers.flatten(out)

        with tf.variable_scope("action_value"):

            action_out = out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        return action_scores

def cnn_to_mlp(convs, hiddens, dueling=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, *args, **kwargs)

def cnn_to_mlp_state_direct(convs, hiddens, dueling=False):

    return lambda *args, **kwargs: _cnn_to_mlp_state_direct(convs, hiddens, dueling, *args, **kwargs)

def cnn_to_mlp_state_embed(convs, hiddens, dueling=False):

    return lambda *args, **kwargs: _cnn_to_mlp_state_embed(convs, hiddens, dueling, *args, **kwargs)

def cnn_to_mlp_symbolic(convs, hiddens):

    return lambda *args, **kwargs: _cnn_to_mlp_symbolic(convs, hiddens, *args, **kwargs)

def cnn_to_mlp_symbolic_multiplex(convs, hiddens):

    return lambda *args, **kwargs: _cnn_to_mlp_symbolic_multiplex(convs, hiddens, *args, **kwargs)

def branch_cnn_to_mlp(convs, hand_hiddens, final_hiddens):

    return lambda *args, **kwargs: _branch_cnn_to_mlp(convs, hand_hiddens, final_hiddens, *args, **kwargs)
