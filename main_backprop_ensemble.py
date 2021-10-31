from typing import *
from batches import split_all, split_subtrain, split_validate, DataSplit, N_ODOR_CLASSES, FEATURE_SIZE
from model_share import get_optimizer_and_scheduler, MAX_EPOCHS
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

def train_epoch(net: nn.Module, criterion, optimizer, T: int, split: DataSplit) -> float:
    """Train the neural network on every available sample in the training data,
    running backpropagation after every instance. For every instance, a random
    context will be sampled.
    Arguments:
        net: a neural network
        criterion: a loss function
        optimizer: a Torch optimizer
        T: onlt batch T will be used for training
        k: the number of samples of each class to use
        split: the set of data to use for training
    Return:
        average_loss: the average loss of all samples in the epoch
    """
    # allow target samples in batches 0...T-1
    sample_range_start = split.batch_ind[T][0]
    sample_range_end = split.batch_ind[T][1]

    # sample_indices[t] is the index of the sample to be processed at step t
    sample_indices = np.arange(sample_range_start, sample_range_end)
    np.random.shuffle(sample_indices)

    # train every sample in the epoch
    running_train_loss = 0.0
    for s in sample_indices:
        x = torch.from_numpy(split.features[s]).type(torch.FloatTensor).unsqueeze(0)
        y = torch.tensor(split.labels[s], dtype=torch.long).unsqueeze(0)

        optimizer.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    return running_train_loss / sample_indices.shape[0]


class NoContextModel(nn.Module):
    def __init__(self):
        super().__init__()

        # skill module
        self.xs = nn.Linear(FEATURE_SIZE, 50)

        # decision module
        self.sd = nn.Linear(50, 20)
        self.dy = nn.Linear(20, N_ODOR_CLASSES)


    def forward(self, x):
        """
        x - shape (batch, input_size)
        :returns: logits
        """
        s = F.relu(self.xs(x))
        d = F.relu(self.sd(s))
        y = self.dy(d)
        return y


class EnsembleModel(nn.Module):
    def __init__(self, classifiers, weights):
        super().__init__()
        self.classifiers = classifiers
        self.weights = weights
    def forward(self, x):
        y_ensemble = None
        for classifier, weight in zip(self.classifiers, self.weights):
            y = classifier(x)
            if y_ensemble is None:
                y_ensemble = torch.zeros_like(y)
            y_ensemble += weight * y
        return y_ensemble


def test_network(net, T: int, split: DataSplit):
    """Returns accuracy of the network on batch T"""
    n_correct = 0
    n_total = 0
    for t in range(*split.batch_ind[T]):
        x = torch.unsqueeze(torch.from_numpy(split.features[t]), 0).type(torch.FloatTensor)
        y = split.labels[t]
        y_hat = net(x).squeeze()

        n_total += 1
        if y == torch.argmax(y_hat).item():
            n_correct += 1

    return float(n_correct) / n_total

def train_nocontext_network(n_epochs: int, T: int):
    net = NoContextModel()

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(net)

    running_loss = 0.0
    lloss = []
    lacc = []
    li = []
    for epoch in range(n_epochs):
        # Adjust the model for one whole epoch
        avg_train_loss = train_epoch(net, criterion, optimizer, T, split_all)
        scheduler.step()

        # Evaluate the accuracy metric
        accuracy = test_network(net, T, split_all)

        # Add all the information to the logs
        lloss.append(avg_train_loss)
        lacc.append(accuracy)
        li.append(epoch)

        print("T=", T, "avg_train_loss=", avg_train_loss, "accuracy=", accuracy, "epoch=", epoch)
    print()

    torch.save(lloss, os.path.join(data_folder, f"lloss_{T}.pt"))
    torch.save(lacc, os.path.join(data_folder, f"lacc_{T}.pt"))
    torch.save(li, os.path.join(data_folder, f"li_{T}.pt"))
    torch.save(net.state_dict(), os.path.join(data_folder, f"model_{T}.pt"))


def evaluate_ensemble(T: int):
    # Ensemble method: Load all networks from t=0...T-1, and estimate weights
    nets = []
    weights = []
    for t in range(T):
        net = NoContextModel()
        net.load_state_dict(torch.load(os.path.join(data_folder, f"model_{t}.pt")))
        nets.append(net)

        acc = test_network(net, T-1, split_all)
        weights.append(acc)

    # Classify each instance
    ensemble = EnsembleModel(nets, weights)
    acc = test_network(ensemble, T, split_all)
    torch.save(acc, os.path.join(data_folder, f"acc_{T}.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    helpstr = "Specify the trial number to save different trial results to different output directories."
    parser.add_argument('-n', '--trial', type=int, default=0, help=helpstr)
    args = parser.parse_args()
    trial = args.trial

    T, trial = trial % 10, trial // 10
    data_folder = f"output/ensemble_short_harddecay{trial}"
    try:
        os.makedirs(data_folder)
    except FileExistsError:
        pass

    # Ensemble method: Train a network for every single batch
    train_nocontext_network(n_epochs=MAX_EPOCHS, T=T)

    if T >= 2:
        import time
        time.sleep(12 * 60 * 60) # 12 hours
        evaluate_ensemble(T=T)

