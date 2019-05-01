import copy as cp
import numpy as np
from sklearn.utils import shuffle
from ..abstraction.transition import Transition


def get_min_and_max_samples_per_class(y):
    """
    Get number of samples for the class with the least and the most samples.
    :param y:       Class indexes for each training sample.
    :return:        Minimum and maximum number of samples.
    """

    min_samples = None
    max_samples = None
    num_classes = np.max(y) + 1

    for class_idx in range(num_classes):

        num_samples = np.sum(y == class_idx)

        if min_samples is None or num_samples < min_samples:
            min_samples = num_samples

        if max_samples is None or num_samples > max_samples:
            max_samples = num_samples

    return min_samples, max_samples


def stratified_split(x, y, validation_fraction, seed=7384):
    """
    Create a stratified training and validation split.
    :param x:       Training data.
    :param y:       Training labels.
    :param validation_fraction:         Validation fraction.
    :return:                            Training and validation splits.
    """

    num_classes = np.max(y) + 1

    #if seed is not None:
    #    np.random.seed(seed)

    x, y = shuffle(x, y)

    valid_x = []
    valid_y = []
    train_x = []
    train_y = []

    for class_idx in range(num_classes):
        mask = y == class_idx

        validation_size = int(np.sum(mask) * validation_fraction)
        assert validation_size > 0

        valid_x.append(x[mask][:validation_size])
        valid_y.append(y[mask][:validation_size])
        train_x.append(x[mask][validation_size:])
        train_y.append(y[mask][validation_size:])

    valid_x = np.concatenate(valid_x, axis=0)
    valid_y = np.concatenate(valid_y, axis=0)
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    return train_x, train_y, valid_x, valid_y


def stratified_cv_split(x, y, num_folds):

    num_classes = np.max(y) + 1
    x, y = shuffle(x, y)

    folds_x = [[] for _ in range(num_folds)]
    folds_y = [[] for _ in range(num_folds)]

    for class_idx in range(num_classes):

        mask = y == class_idx
        fold_size = int(np.sum(mask) / num_folds)

        for fold_idx in range(num_folds):

            assert fold_size > 0

            if fold_idx == num_folds - 1:
                folds_x[fold_idx].append(x[mask][fold_idx * fold_size:])
                folds_y[fold_idx].append(y[mask][fold_idx * fold_size:])
            else:
                folds_x[fold_idx].append(x[mask][fold_idx * fold_size : (fold_idx + 1) * fold_size])
                folds_y[fold_idx].append(y[mask][fold_idx * fold_size : (fold_idx + 1) * fold_size])

    folds_x = [np.concatenate(folds_x[i], axis=0) for i in range(num_folds)]
    folds_y = [np.concatenate(folds_y[i], axis=0) for i in range(num_folds)]

    return folds_x, folds_y


def undersample(x, y, factor=None, target=None):
    """
    Undersample the data.
    :param x:           Data.
    :param y:           Labels.
    :param factor:      How many times less data is allowed.
    :return:            Undersampled data.
    """

    # either provide factor or target
    assert factor is None or target is None
    assert factor is not None or target is not None

    if factor is not None:
        min_samples, max_samples = get_min_and_max_samples_per_class(y)
        target = min(min_samples * factor, max_samples)

    num_classes = np.max(y) + 1

    x, y = shuffle(x, y)

    sampled_x = []
    sampled_y = []

    for class_idx in range(num_classes):

        mask = y == class_idx

        sampled_x.append(x[mask][:target])
        sampled_y.append(y[mask][:target])

    sampled_x = np.concatenate(sampled_x, axis=0)
    sampled_y = np.concatenate(sampled_y, axis=0)

    return shuffle(sampled_x, sampled_y)


def oversample(x, y):
    """
    Oversample the data.
    :param x:       Data.
    :param y:       Labels.
    :return:        Oversampled data.
    """

    _, max_samples = get_min_and_max_samples_per_class(y)
    classes = np.unique(y)

    x, y = shuffle(x, y)

    sampled_x = []
    sampled_y = []

    for class_idx in classes:

        mask = y == class_idx

        if np.sum(mask) == max_samples:

            sampled_x.append(x[mask])
            sampled_y.append(y[mask])

        else:

            sampled_x.append(np.random.choice(x[mask], size=max_samples, replace=True))
            sampled_y.append(np.random.choice(y[mask], size=max_samples, replace=True))

    sampled_x = np.concatenate(sampled_x, axis=0)
    sampled_y = np.concatenate(sampled_y, axis=0)

    return shuffle(sampled_x, sampled_y)


def flip_classes(labels, fraction):

    labels = cp.deepcopy(labels)

    if fraction == 0.0:
        return labels

    classes = list(np.unique(labels))

    for cls in classes:

        all_indices = np.where(labels == cls)[0]
        flip_indices = np.random.choice(all_indices, size=int(len(all_indices) * fraction), replace=False)

        tmp_classes = cp.deepcopy(classes)
        tmp_classes.remove(cls)
        new_classes = np.random.choice(tmp_classes, size=len(flip_indices), replace=True)

        labels[flip_indices] = new_classes

    return labels


def get_experience_from_replay_buffer(replay_buffer, limit=None):

    buffer_size = len(replay_buffer._storage)

    if limit is not None:
        num_exp = min(buffer_size, limit)
    else:
        num_exp = buffer_size

    transitions = []
    end_idx = (replay_buffer._next_idx - 1) % buffer_size

    for idx in range(num_exp):

        idx_mod = (end_idx - idx) % buffer_size
        data = replay_buffer._storage[idx_mod]

        t = Transition(data[0], data[1], data[2], data[3], False, data[4])
        transitions.append(t)

    # the oldest should be the first
    transitions = list(reversed(transitions))

    return transitions
