"""Train, test, and validation batches, plus some helpful indexes.
This module exposes *singleton objects*, so changing the values of a vector
may have unintended side effects."""

from preprocess import load_features_labels
import numpy as np
from enum import Enum
from typing import *

features, labels, batch_ind = load_features_labels(include_gas_6=False)

# label_set - an ndarray of shape (n_labels)
label_set = np.unique(labels)

N_ODOR_CLASSES = label_set.shape[0]
FEATURE_SIZE = features.shape[1]

def make_batch_label_index(batch_ind_, labels_):
    samples_in_batch_by_label = []
    for start, end in batch_ind_:
        Y = labels_[start:end]
        samples_by_label = []
        for label in label_set:
            matching_indices, = np.where(Y==label)
            samples_by_label.append(matching_indices + start)
        samples_in_batch_by_label.append(samples_by_label)
    return samples_in_batch_by_label

samples_in_batch_by_label = make_batch_label_index(batch_ind, labels)

###############################################################################
# Divide the data into subtrain / validation sets.
###############################################################################
validate_portion = 0.5

# Sample a balanced set for each batch
batch_ind_r = []
batch_ind_v = []

indices_r = np.empty(shape=0, dtype=np.long)
indices_v = np.empty(shape=0, dtype=np.long)

for batch in range(len(batch_ind)):

    start_r = indices_r.shape[0]
    start_v = indices_v.shape[0]

    for label in label_set:
        samps = samples_in_batch_by_label[batch][label]
        n_samps = samps.shape[0]
        n_validate = int(validate_portion * n_samps)
        validate_samps = np.random.choice(samps, size=n_validate, replace=False)
        subtrain_samps = np.setdiff1d(samps, validate_samps, assume_unique=True)

        indices_r = np.concatenate((indices_r, subtrain_samps))
        indices_v = np.concatenate((indices_v, validate_samps))

    batch_ind_r.append((start_r, indices_r.shape[0]))
    batch_ind_v.append((start_v, indices_v.shape[0]))

labels_r = labels[indices_r]
features_r = features[indices_r]
labels_v = labels[indices_v]
features_v = features[indices_v]
samples_in_batch_by_label_r = make_batch_label_index(batch_ind_r, labels_r)
samples_in_batch_by_label_v = make_batch_label_index(batch_ind_v, labels_v)

###############################################################################
# Handy data class interface organizing the different splits
###############################################################################
class DataSplit:
    """A dataclass representing a subset of data.

    Attributes:
        features - an ndarray of shape (n_samples, n_features)
        labels - an ndarray of shape (n_samples,) and values ranging from 0 to n_labels-1
        batch_ind - a list of tuples (batch_start, batch_end) which are valid slice indices
        samples_in_batch_by_label[batch][label] - an ndarray of shape (n,)
                containing a indices into the *features* / *labels* arrays,
                of samples matching the label *label* within the batch *batch*,
                for *batch* in 0..9 and *label* in 0..4.
    """
    features: np.ndarray
    labels: np.ndarray
    batch_ind: List[Tuple[int]]
    samples_in_batch_by_label: List[List[np.ndarray]]

    def __init__(self, features, labels, batch_ind, samples_in_batch_by_label):
        self.features = features
        self.labels = labels
        self.batch_ind = batch_ind
        self.samples_in_batch_by_label = samples_in_batch_by_label

split_all = DataSplit(features=features,
    labels=labels,
    batch_ind=batch_ind,
    samples_in_batch_by_label=samples_in_batch_by_label)
split_subtrain = DataSplit(features=features_r,
    labels=labels_r,
    batch_ind=batch_ind_r,
    samples_in_batch_by_label=samples_in_batch_by_label_r)
split_validate = DataSplit(features=features_v,
    labels=labels_v,
    batch_ind=batch_ind_v,
    samples_in_batch_by_label=samples_in_batch_by_label_v)
