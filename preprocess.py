#!/usr/bin/env python3
"""Data preprocessing of the Gas Sensor Array Drift Dataset Data Set:
    https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset
This file may be executed to preprocess the data and save the result to files.
"""

import numpy as np
import os
import re
import pickle

ODOR_DATA_DIR = "Dataset"
PREPROCESSED_DATA_PATH = os.path.join(ODOR_DATA_DIR, "dataset_preprocessed.pkl")
PREPROCESSED_DATA_PATH_NO_6 = os.path.join(ODOR_DATA_DIR, "dataset_preprocessed_no_6.pkl")

def _load_raw_data(include_gas_6):
    # Load each batch, one by one, into a dictionary
    batch_pattern = re.compile("batch(\d+)\.dat")
    batches_d = {}
    for entry in os.scandir(ODOR_DATA_DIR):
        match = batch_pattern.match(entry.name)
        if match is None:
            continue
        batch_num = int(match.group(1))-1

        # Read the file line by line
        feats = []
        labels = []
        with open(entry.path, "r") as fp:
            for line in fp.readlines():
                parts = line.split(" ")
                label = int(parts[0])-1
                if include_gas_6 == False and label == 5:
                    continue
                labels.append(label)
                feat = []
                feats.append(feat)
                for i, part in enumerate(parts[1:]):
                    feat_label, value = part.split(":")
                    feat_num = int(feat_label)-1
                    assert(feat_num==i)
                    val = float(value)
                    feat.append(val)
        batches_d[batch_num] = (feats, labels)

    # Combine all batches into a single object and create an index
    batch_ind = []
    last_batch_i = max(batches_d.keys())
    all_features = []
    all_labels = []
    k = 0
    for batch_i in range(0, last_batch_i+1):
        feats, labels = batches_d[batch_i]
        all_features.append(np.array(feats))
        all_labels.append(np.array(labels))
        batch_ind.append((k, k+len(feats)))
        k+=len(feats)

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)

    return features, labels, batch_ind

def preprocess_and_save(include_gas_6=True):
    """Read all data in the gas odor sensor dataset, then:
    1. Format all data into numpy arrays
    2. z-score the features
    """
    features, labels, batch_ind = _load_raw_data(include_gas_6)

    # z-score all features along the sample axis, so that each feature has the same weight.
    import scipy.stats as stats
    features_z = stats.zscore(features, axis=0)

    if include_gas_6:
        pickle.dump((features_z, labels, batch_ind), open(PREPROCESSED_DATA_PATH, "wb"))
    else:
        pickle.dump((features_z, labels, batch_ind), open(PREPROCESSED_DATA_PATH_NO_6, "wb"))


def load_features_labels(include_gas_6=True):
    """
    returns a tuple (features, labels, batch_ind), where:
        features - an ndarray of shape (n_samples, n_features)
        labels - an ndarray of shape (n_samples,) and values ranging from 0 to n_labels-1
        batch_ind - a list of tuples (batch_start, batch_end) which are valid slice indices
    """
    if include_gas_6:
        return pickle.load(open(PREPROCESSED_DATA_PATH, "rb"))
    else:
        return pickle.load(open(PREPROCESSED_DATA_PATH_NO_6, "rb"))

if __name__ == "__main__":
    preprocess_and_save()
    preprocess_and_save(include_gas_6=False)
    features, labels, batch_ind = load_features_labels()
    assert(features.shape[0]==labels.shape[0])
    assert(batch_ind[-1][1]==features.shape[0])
    import ipdb; ipdb.set_trace()