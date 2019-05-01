import collections
import copy as cp
import numpy as np
from .transition import Transition


def collect_experience(env, act, get_block, buffers, min_buffer_size=None, max_buffer_size=None, num_episodes=None,
                       vectorized_state=False, verbose=False, keys=None, start_timestep=0, task=None,
                       saver=None, no_dedup=False):
    """
    Collect unique experience.
    :param env:                 Environment to act in.
    :param act:                 Act function.
    :param get_block:           Get state-action block function.
    :param buffers:             List of state-action block experience buffers.
    :param min_buffer_size:     Minimum buffer size, the loop doesn't end before this is reached.
    :param max_buffer_size:     Maximum buffer size.
    :param num_episodes:        Number of episodes to run for. If min_buffer_size is not reached, run longer.
    :param vectorized_state:    The state coming from the environment is already vectorized.
    :param verbose:             Print more information.
    :param task:                Current task ID.
    :return:                    Per-episode rewards.
    """

    assert min_buffer_size is not None or num_episodes is not None

    rewards = []
    episode_rewards = []
    if keys is None:
        keys = set()
    is_start = True

    timestep = start_timestep
    while True:

        # check if end
        if (num_episodes is None or len(rewards) >= num_episodes) and \
                (min_buffer_size is None or np.all([len(buffer) >= min_buffer_size for buffer in buffers])):
            break

        state = cp.deepcopy(env.get_state())

        if saver is not None:
            saver.add(state[0][:, :, 0])

        _, length_to_goal, _, action, next_state, _, _, _, reward, done = act(state, timestep)

        episode_rewards.append(reward)

        next_state = cp.deepcopy(next_state)
        block = get_block(state, action, reward)

        transition = Transition(state, action, reward, next_state, is_start, done, task=task)

        if no_dedup:

            if max_buffer_size is not None and len(buffers[block]) >= max_buffer_size:
                del buffers[block][0]

            buffers[block].append(transition)

        else:

            if vectorized_state:
                key = tuple(state) + (action,)
            else:
                key = tuple(state[0].reshape(-1, ).tolist()) + (state[1], action)

            if key not in keys:
                # unique experience found
                keys.add(key)

                if max_buffer_size is not None and len(buffers[block]) >= max_buffer_size:
                    # buffer is overflowing, delete the oldest experience
                    if vectorized_state:
                        key = tuple(buffers[block][0].state) + (buffers[block][0].action,)
                    else:
                        key = tuple(buffers[block][0].state[0].reshape(-1, ).tolist()) + (buffers[block][0].state[1],
                                                                                          buffers[block][0].action)

                    try:
                        keys.remove(key)
                    except KeyError:
                        continue

                    del buffers[block][0]
                buffers[block].append(transition)

        is_start = False

        if done:
            # episode ended
            if saver is not None:
                saver.add(next_state[0][:, :, 0])
                saver.save_gif()
                saver.reset()

            rewards.append(np.sum(episode_rewards))
            episode_rewards = []
            env.reset()
            is_start = True

        timestep += 1

    return rewards, timestep


def collect_experience_option(env, act, get_block, buffers, option, min_buffer_size=None, max_buffer_size=None,
                              num_episodes=None, vectorized_state=False, verbose=False, keys=None, start_timestep=0,
                              task=None, saver=None, remember=None):

    assert min_buffer_size is not None or num_episodes is not None

    rewards = []
    episode_rewards = []
    if keys is None:
        keys = set()
    is_start = True
    option_done = False

    timestep = start_timestep
    while True:

        state = cp.deepcopy(env.get_state())

        if saver is not None:
            saver.add(state[0][:, :, 0])

        if not option_done:
            # take a step using the option
            _, steps_to_goal, _, action, next_state, _, _, _, reward, done = option(state, timestep)
            next_state = cp.deepcopy(next_state)

            if steps_to_goal == 0:
                option_done = True

            if remember is not None:
                remember(state, action, reward, next_state, done)
        else:
            # take step using the policy
            _, _, _, action, next_state, _, _, _, reward, done = act(state, timestep)
            next_state = cp.deepcopy(next_state)

        episode_rewards.append(reward)

        next_state = cp.deepcopy(next_state)
        block = get_block(state, action, reward)

        if vectorized_state:
            key = tuple(state) + (action,)
        else:
            key = tuple(state[0].reshape(-1, ).tolist()) + (state[1], action)

        if key not in keys:
            # unique experience found
            keys.add(key)

            transition = Transition(state, action, reward, next_state, is_start, done, task=task)

            if max_buffer_size is not None and len(buffers[block]) >= max_buffer_size:
                # buffer is overflowing, delete the oldest experience
                if vectorized_state:
                    key = tuple(buffers[block][0].state) + (buffers[block][0].action,)
                else:
                    key = tuple(buffers[block][0].state[0].reshape(-1, ).tolist()) + (buffers[block][0].state[1],
                                                                                      buffers[block][0].action)

                try:
                    keys.remove(key)
                except KeyError:
                    continue

                del buffers[block][0]
            buffers[block].append(transition)

        # check if end
        if (num_episodes is None or len(rewards) >= num_episodes) and \
                (min_buffer_size is None or np.all([len(buffer) >= min_buffer_size for buffer in buffers])):
            break
        elif verbose:
            for idx, buffer in enumerate(buffers):
                print("buffer {:d} len: {:d}".format(idx + 1, len(buffer)))

        is_start = False

        if done:
            # episode ended
            if saver is not None:
                saver.add(next_state[0][:, :, 0])
                saver.save_gif()
                saver.reset()

            rewards.append(np.sum(episode_rewards))
            episode_rewards = []
            env.reset()
            is_start = True
            option_done = False

        timestep += 1

    return rewards, timestep


def collect_experience_q_option(env, act, get_block, buffers, option, option_agent, discount, epsilon,
                                min_buffer_size=None, max_buffer_size=None, num_episodes=None, vectorized_state=False,
                                verbose=False, keys=None, start_timestep=0, task=None, saver=None, remember=None):

    assert min_buffer_size is not None or num_episodes is not None

    rewards = []
    episode_rewards = []
    if keys is None:
        keys = set()
    is_start = True
    option_done = False
    goal = None

    timestep = start_timestep
    while True:

        state = cp.deepcopy(env.get_state())

        if saver is not None:
            saver.add(state[0][:, :, 0])

        if not option_done:

            if is_start:
                goal = option_agent.act_e_greedy(epsilon)

            # take a step using the option
            _, steps_to_goal, _, action, next_state, _, _, _, reward, done = option(state, timestep)
            next_state = cp.deepcopy(next_state)

            if steps_to_goal == 0:
                option_done = True

            if remember is not None:
                remember(state, action, reward, next_state, done)
        else:
            # take step using the policy
            _, _, _, action, next_state, _, _, _, reward, done = act(state, timestep)
            next_state = cp.deepcopy(next_state)

        episode_rewards.append(reward)

        block = get_block(state, action, reward)

        if vectorized_state:
            key = tuple(state) + (action,)
        else:
            key = tuple(state[0].reshape(-1, ).tolist()) + (state[1], action)

        if key not in keys:
            # unique experience found
            keys.add(key)

            transition = Transition(state, action, reward, next_state, is_start, done, task=task)

            if max_buffer_size is not None and len(buffers[block]) >= max_buffer_size:
                # buffer is overflowing, delete the oldest experience
                if vectorized_state:
                    key = tuple(buffers[block][0].state) + (buffers[block][0].action,)
                else:
                    key = tuple(buffers[block][0].state[0].reshape(-1, ).tolist()) + (buffers[block][0].state[1],
                                                                                      buffers[block][0].action)

                try:
                    keys.remove(key)
                except KeyError:
                    continue

                del buffers[block][0]
            buffers[block].append(transition)

        # check if end
        if (num_episodes is None or len(rewards) >= num_episodes) and \
                (min_buffer_size is None or np.all([len(buffer) >= min_buffer_size for buffer in buffers])):
            break
        elif verbose:
            for idx, buffer in enumerate(buffers):
                print("buffer {:d} len: {:d}".format(idx + 1, len(buffer)))

        is_start = False

        if done:
            # episode ended

            # maybe save the episode as a gif
            if saver is not None:
                saver.add(next_state[0][:, :, 0])
                saver.save_gif()
                saver.reset()

            # update options q-values
            return_val = np.sum([(discount ** i) * r for i, r in enumerate(episode_rewards)])
            option_agent.learn(goal, return_val)

            # save rewards and reset the environment
            rewards.append(np.sum(episode_rewards))
            episode_rewards = []
            env.reset()
            is_start = True
            option_done = False

        timestep += 1

    return rewards, timestep


def collect_experience_dictionary(env, act, experience, num_episodes=None, num_timesteps=None,
                                  start_timestep=0, task=None):

    assert num_episodes is not None or num_timesteps is not None

    # initialize containers
    rewards = []
    episode_rewards = []
    num_rewards = 0

    # remember if this is the first transition of the episode
    is_start = True

    timestep = start_timestep
    while True:

        # act
        state = cp.deepcopy(env.get_state())
        _, _, _, action, next_state, _, _, _, reward, done = act(state, timestep)

        # save rewards
        episode_rewards.append(reward)

        next_state = cp.deepcopy(next_state)

        # save experience
        if reward > 0:
            num_rewards += 1

        key = tuple(state) + (action,)
        transition = Transition(state, action, reward, next_state, is_start, done, task=task)

        # if reward > 0 or key not in experience or experience[key].reward <= 0:
        experience[key] = transition

        # check if end
        if num_episodes is not None and len(rewards) >= num_episodes:
            break

        if num_timesteps is not None and timestep - start_timestep >= num_timesteps:
            break

        is_start = False

        if done:
            # episode ended
            rewards.append(np.sum(episode_rewards))
            episode_rewards = []
            env.reset()
            is_start = True

        timestep += 1

    return rewards, timestep, num_rewards


def reclassify_experience(experience, get_block, num_blocks):
    """
    Classify experience into state-action blocks.
    :param experience:      List of experienced transitions.
    :param get_block:       Get state-action block function.
    :param num_blocks:      Number of state-action blocks.
    :return:
    """

    buffers = [[] for _ in range(num_blocks)]

    for transition in experience:
        block = get_block(transition.state, transition.action, transition.reward)
        buffers[block].append(transition)

    return buffers


def run_value_iteration(state_action_successor_table, state_successor_table, gamma, get_reward, diff_threshold,
                        max_steps=1000):
    """
    Run q-value iteration over a state-action partition.
    :param state_action_successor_table:        State-action successor table.
    :param state_successor_table:               State successor table.
    :param gamma:                               Discount factor.
    :param get_reward:                          Get reward function.
    :param diff_threshold:                      Termination threshold.
    :param max_steps:                           Maximum number of value iteration steps.
    :return:                                    Q-values.
    """

    # create dictionaries for rewards and q-values
    q = {key: 0 for key in state_action_successor_table.keys()}
    r = {key: get_reward(key, value is None) for key, value in state_action_successor_table.items()}

    # run q-value iteration
    for _ in range(max_steps):

        # compute new q values
        new_q = {key: 0 for key in state_action_successor_table.keys()}

        for action_key, state_key in state_action_successor_table.items():

            if state_key is None:
                new_q[action_key] = r[action_key]
            else:
                next_values = []
                for next_key in state_successor_table[state_key]:
                    next_values.append(q[next_key])

                new_q[action_key] = r[action_key] + gamma * np.max(next_values)

        # check if the max difference between new and old q values is below termination threshold
        max_diff = None
        for key in state_action_successor_table.keys():
            diff = np.abs(q[key] - new_q[key])
            if max_diff is None or diff > max_diff:
                max_diff = diff

        if max_diff < diff_threshold:
            break

        # set new q-values
        q = new_q

    return q


def get_reward_num_samples(block_index, goal, state_action_partition, max_threshold, goal_reward):
    """
    Give rewards for state-action blocks that have less samples than a threshold.
    :param block_index:                     Index of the state-action block.
    :param goal:                            True if this state-action block transitions into a goal state.
    :param state_action_partition:          State-action partition.
    :param max_threshold:                   Max samples.
    :param goal_reward:                     Goal state-action reward. The higher, the less exploration.
    :return:                                Reward.
    """

    if goal:
        # goal state-action
        return goal_reward
    else:
        # calculate reward based on how interesting a particular state-action block is
        return 1 - (min(len(state_action_partition.get(block_index)), max_threshold) / max_threshold)


def get_reward_confidence(block_index, goal, state_action_partition, max_threshold, goal_reward, min_confidence):
    """
    Give rewards for state-action blocks that have confidence less than a threshold.
    :param block_index:                     Index of the state-action block.
    :param goal:                            True if this state-action block transitions into a goal state.
    :param state_action_partition:          State-action partition.
    :param max_threshold:                   Max confidence.
    :param goal_reward:                     Goal state-action reward. The higher, the less exploration.
    :param min_confidence:                  Take the minimum transition confidence instead of the mean.
    :return:                                Reward.
    """

    if goal:
        # goal state-action
        return goal_reward
    else:
        # calculate reward based on how interesting a particular state-action block is
        confidences = []

        for transition in state_action_partition.get(block_index):
            confidences.append(transition.next_state_block_confidence)

        if min_confidence:
            confidence = np.min(confidences)
        else:
            confidence = np.mean(confidences)

    return 1 - (min(confidence, max_threshold) / max_threshold)


def get_next_state_action_blocks_2(next_state, classifier, sample_actions, confidence_threshold=0.0,
                                   confidence_percentile=False, exclude_blocks=False, block_confs=None):

    sampled_actions = sample_actions(next_state)

    predictions = classifier.predict(
        [next_state for _ in range(len(sampled_actions))], sampled_actions
    )

    probs = []
    blocks = set()

    for prediction in predictions:

        block = np.argmax(prediction)
        prob = prediction[block]

        if block_confs is not None:
            prob *= block_confs[block]

        probs.append(prob)

        # add the block, but only under some conditions
        if not exclude_blocks or prob >= confidence_threshold:
            blocks.add(block)

    # check if the system if confident enough about the state
    if confidence_percentile is not None:
        confidence = np.percentile(probs, confidence_percentile)
    else:
        confidence = np.min(probs)

    return frozenset(blocks), confidence


def get_next_state_action_blocks_3(next_state, classifier, confidence_threshold=0.0, confidence_percentile=False,
                                   exclude_blocks=False, block_confs=None):

    predictions = classifier.predict_all_actions(next_state)

    probs = []
    blocks = set()

    for prediction in predictions[0]:
        block = np.argmax(prediction)
        prob = prediction[block]

        if block_confs is not None:
            prob *= block_confs[block]

        probs.append(prob)

        # add the block, but only under some conditions
        if not exclude_blocks or prob >= confidence_threshold:
            blocks.add(block)

    # check if the system if confident enough about the state
    if confidence_percentile is not None:
        confidence = np.percentile(probs, confidence_percentile)
    else:
        confidence = np.min(probs)

    return frozenset(blocks), confidence


def get_next_state_action_blocks(next_state, classifier, sample_actions, confidence_threshold=0.0,
                                 nth_min_confidence=False, confidence_percentile=False, confidence_product=False):
    """
    Get all available state-action blocks for the next state.
    :param next_state:                  Next state.
    :param classifier:                  State-action block classifier.
    :param sample_actions:              Function that samples actions for a given state.
    :param confidence_threshold:        Threshold on the classifier confidence.
    :param nth_min_confidence:          Aggregate confidence using the nth minimum function.
    :param confidence_product:          Aggregate confidence using a product.
    :return:                            Set of next available state-action blocks and their aggregate confidence.
    """

    assert not (nth_min_confidence is not None and confidence_product)

    actions = sample_actions(next_state)

    predictions = classifier.predict([next_state] * len(actions), actions)
    max_confidence = np.max(predictions, axis=1)
    above_threshold = max_confidence > confidence_threshold

    if not np.any(above_threshold):
        above_threshold[np.argmax(max_confidence)] = True

    blocks = np.argmax(predictions, axis=1)[above_threshold]
    blocks = set(blocks)

    if nth_min_confidence is not None:
        if nth_min_confidence == 1:
            confidence = np.min(max_confidence[above_threshold])
        else:
            confidence = np.partition(max_confidence[above_threshold], nth_min_confidence)[nth_min_confidence]
    elif confidence_percentile is not None:
        confidence = np.percentile(max_confidence, confidence_percentile)
    elif confidence_product:
        confidence = np.product(max_confidence[above_threshold])
    else:
        confidence = np.mean(max_confidence[above_threshold])

    return blocks, confidence


def sort_transferred_experience(experience):
    """
    Sort experience based on tasks and rewards.
    :param experience:      List of experienced transitions.
    :return:                List of buffers for different kinds of transitions.
                            The first buffer is for all transitions without rewards,
                            the second is an empty buffer for new experience and the rest of the buffers
                            contain transitions with rewards from old tasks.
    """

    buffer_dict = collections.defaultdict(list)

    for transition in experience:
        if transition.reward > 0:
            key = (transition.reward, transition.task)
        else:
            key = None

        buffer_dict[key].append(transition)

    buffers = []

    # the first buffer is for experience without rewards
    if None in buffer_dict:
        buffers.append(buffer_dict[None])
    else:
        buffers.append([])

    # the second buffer is for new experience
    buffers.append([])

    # the rest is for transitions with rewards from previous tasks
    for key, value in buffer_dict.items():
        if key is not None:
            buffers.append(value)

    return buffers


def delete_experience(experience, keys):

    for transition in experience:
        key = tuple(transition.state[0].reshape(-1, ).tolist()) + (transition.state[1], transition.action)
        keys.remove(key)


def update_mean(value, mean, count):
    """
    Update value of a streaming mean.
    :param value:     New value.
    :param mean:      Mean value.
    :param count:     Number of values averaged.
    :return:
    """

    return (value - mean) / (count + 1)


def softmax(x, t):
    """
    Softmax function.
    :param x:   Array of values.
    :param t:   Temperature.
    :return:    Softmax distribution.
    """

    e_x = np.exp(x / t)
    return e_x / e_x.sum()


def simulate_state_block_probability(confusion_matrix, predictions, block_indices, num_sims=10000):
    """
    Simulate the probability of a state-action block based on the confusion matrix.
    :param confusion_matrix:        Confusion matrix for a classifier.
    :param predictions:             Predictions by the classifier.
    :param block_indices:           Target block indices.
    :param num_sims:                Number of simulations.
    :return:                        Probability of the state-action block.
    """

    block_indices = set(block_indices)
    num_blocks = confusion_matrix.shape[0]

    sims = []
    for prediction in predictions:
        sim = np.random.choice(list(range(num_blocks)), size=num_sims, replace=True,
                               p=confusion_matrix[prediction, :])
        sims.append(sim)
    sims = np.stack(sims, axis=0)

    mask = np.apply_along_axis(lambda x: set(x) == block_indices, 0, sims)

    return np.sum(mask) / num_sims


def get_gt_distribution(experience, get_block):
    """
    Get the ground-truth distribution of an experience buffer.
    :param experience:      List of transitions.
    :param get_block:       Function that returns the ground-truth state-action block for each transition.
    :return:                Number of samples for each state-action block.
    """

    dist = collections.defaultdict(lambda: 0)

    for transition in experience:
        key = get_block(transition.state, transition.reward, transition.next_state)
        dist[key] += 1

    dist = dict(dist)
    return dist


def deduplicate(experience, vectorized_state=False):

    keys = set()
    original = []

    for transition in experience:

        if vectorized_state:
            key = tuple(transition.state) + (transition.action,)
        else:
            key = tuple(transition.state[0].reshape(-1, ).tolist()) + (transition.state[1], transition.action)

        if key not in keys:
            keys.add(key)
            original.append(transition)

    return original


def evaluate_bisimulation(embeddings, labels, r_hat=None, p_hat=None, r=None, p=None, mask_goal=True):

    assert embeddings.shape[0] == labels.shape[0]

    assignment = np.argmax(embeddings, axis=1)
    matches = np.zeros((embeddings.shape[1], embeddings.shape[1]))

    for i in range(embeddings.shape[1]):
        for j in range(embeddings.shape[1]):
            matches[i, j] = np.sum(np.logical_and(assignment == i, labels == j))

    correct = 0
    assigned_labels = {}

    for i in range(embeddings.shape[1]):

        max_coords = np.unravel_index(matches.argmax(), matches.shape)

        assigned_labels[max_coords[1]] = max_coords[0]

        correct += matches[max_coords]

        matches[max_coords[0], :] = -1
        matches[:, max_coords[1]] = -1

    accuracy = correct / embeddings.shape[0]

    if r_hat is not None and p_hat is not None and r is not None and p is not None:

        perm = [assigned_labels[i] for i in sorted(assigned_labels.keys())]

        r_hat = np.stack([
            r_hat[a][perm] for a in range(r_hat.shape[0])
        ], axis=0)

        p_hat = np.stack([
            p_hat[a][perm][:, perm] for a in range(p_hat.shape[0])
        ], axis=0)

        if mask_goal:
            goals = r_hat > 0.8
            p_hat[goals, :] = 0.0

        return accuracy, np.sum(np.square(r_hat - r)), np.sum(np.square(p_hat - p))

    else:

        return accuracy, None, None
