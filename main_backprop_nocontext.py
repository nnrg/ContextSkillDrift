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
        T: batches up to but not including T will be used for training
        k: the number of samples of each class to use
        split: the set of data to use for training
    Return:
        average_loss: the average loss of all samples in the epoch
    """
    # allow target samples in batches 0...T-1
    sample_range_start = 0
    sample_range_end = split.batch_ind[T-1][1]

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


def get_loss_epoch(net: nn.Module, criterion, T: int, split: DataSplit) -> float:
    """Train the neural network on every available sample in the training data,
    running backpropagation after every instance. For every instance, a random
    context will be sampled.
    Arguments:
        net: a neural network
        criterion: a loss function
        optimizer: a Torch optimizer
        T: batches up to but not including T will be used for training
        k: the number of samples of each class to use
        split: the set of data to use for training
    Return:
        average_loss: the average loss of all samples in the epoch
    """
    # allow target samples in batches 0...T-1
    sample_range_start = 0
    sample_range_end = split.batch_ind[T-1][1]

    # sample_indices[t] is the index of the sample to be processed at step t
    sample_indices = np.arange(sample_range_start, sample_range_end)
    np.random.shuffle(sample_indices)

    # train every sample in the epoch
    running_train_loss = 0.0
    for s in sample_indices:
        x = torch.from_numpy(split.features[s]).type(torch.FloatTensor).unsqueeze(0)
        y = torch.tensor(split.labels[s], dtype=torch.long).unsqueeze(0)

        y_pred = net(x)
        loss = criterion(y_pred, y)

        running_train_loss += loss.item()

    return running_train_loss / sample_indices.shape[0]


def calculate_stopping_time(patience: int, T: int):
    """Early stopping implementation.
    Train the network, evaluating its accuracy on the validation set at
    fixed intervals. If its accuracy dropped consecutively for a number of
    times equal to the patience parameter, then terminate training.

    Arguments:
        patience: the number of times the algorithm will tolerate a worse
            accuracy on the validation set before terminating
        T: the testing batch (model will be trained on batches 0...T-1
    Saved outputs:
        val_lacc_{T}.pt: a list of sampled accuracies on the validation set
        val_loss_{T}.pt: a list of sampled losses on the train set
        val_li_{T}.pt: the training sample at which the above were evaluated
        val_stop_time_{T}.pt: a single integer, the stop time"""

    net = NoContextModel()
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(net)

    lloss = []
    lacc = []
    li = []
    consecutive_worse_count = 0 #  number consecutive evaluations with decreasing accuracy

    validation_loss_prev = float('inf')

    stopping_epoch = 0
    epoch = 0

    while consecutive_worse_count < patience and epoch < 100:
        # Adjust the model for one whole epoch
        avg_train_loss = train_epoch(net, criterion, optimizer, T, split_subtrain)
        scheduler.step()

        # Evaluate the network on the validation set
        validation_loss = get_loss_epoch(net, criterion, T, split_validate)

        # Update the patience parameter
        if validation_loss < validation_loss_prev:
            consecutive_worse_count = 0
            validation_loss_prev = validation_loss
            stopping_epoch = epoch
        else:
            consecutive_worse_count += 1

        # Add all the information to the logs
        lloss.append(avg_train_loss)
        lacc.append(validation_loss)
        li.append(epoch)

        epoch += 1

    torch.save(lloss, os.path.join(data_folder, f"val_lloss_{T}.pt"))
    torch.save(lacc, os.path.join(data_folder, f"val_lacc_{T}.pt"))
    torch.save(li, os.path.join(data_folder, f"val_li_{T}.pt"))
    torch.save(stopping_epoch, os.path.join(data_folder, f"val_stop_time_{T}.pt"))

    return stopping_epoch


class NoContextModel(nn.Module):
    def __init__(self, skill_size=50, decision_size=20):
        super().__init__()

        # skill module
        self.xs = nn.Linear(FEATURE_SIZE, skill_size)

        # decision module
        self.sd = nn.Linear(skill_size, decision_size)
        self.dy = nn.Linear(decision_size, N_ODOR_CLASSES)


    def forward(self, x):
        """
        x - shape (batch, input_size)
        :returns: logits
        """
        s = F.relu(self.xs(x))
        d = F.relu(self.sd(s))
        y = self.dy(d)
        return y

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

        # Evaluate the network on the validation set
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


def evaluate_network(T: int):
    net = NoContextModel()
    net.load_state_dict(torch.load(os.path.join(data_folder, f"model_{T}.pt")))
    acc = test_network(net, T, split_all)
    torch.save(acc, os.path.join(data_folder, f"acc_{T}.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    helpstr = "Specify the trial number to save different trial results to different output directories."
    parser.add_argument('-n', '--trial', type=int, default=0, help=helpstr)
    args = parser.parse_args()
    trial = args.trial

    T, trial = trial % 10, trial // 10
    data_folder = f"output/nocontext_short_harddecay{trial}"
    try:
        os.makedirs(data_folder)
    except FileExistsError:
        pass

    if T >= 1:
        # stopping_time = calculate_stopping_time(patience=10, T=T)
        # print("T=", T, "stopping_time=", stopping_time)
        train_nocontext_network(n_epochs=MAX_EPOCHS, T=T)

    if T >= 1:
        evaluate_network(T=T)

