from batches import split_all, split_subtrain, split_validate, DataSplit, N_ODOR_CLASSES, FEATURE_SIZE
from model_share import get_optimizer_and_scheduler, MAX_EPOCHS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys

def choose_k(k: int, batch_num: int, split: DataSplit) -> np.ndarray:
    """Sampling method for the batch context.

    Arguments:
        k: The number of odors of each class to sample
        batch_num: The batch to sample from, 0 thru 9
    Return: A numpy array of shape (N_ODOR_CLASSES*FEATURE_SIZE*k,)
        assuming there are 5 classes and 128 feature dimensions.
        The first 5*128 dimensions correspond to the first class, and so on."""
    indices = []
    for c in range(N_ODOR_CLASSES):
        indices += np.random.choice(
            split.samples_in_batch_by_label[batch_num][c],
            size=k,
            replace=False).tolist()
    return np.concatenate(split.features[indices])


class ContextModel(nn.Module):
    def __init__(self, k: int):
        super().__init__()

        # context module
        self.lc = nn.RNN(k * N_ODOR_CLASSES * FEATURE_SIZE, 10, nonlinearity='relu')

        # skill module
        self.xs = nn.Linear(FEATURE_SIZE, 50)

        # decision module
        self.cd = nn.Linear(10, 20)
        self.sd = nn.Linear(50, 20)
        self.dy = nn.Linear(20, N_ODOR_CLASSES)

    def forward(self, ll, x):
        """
        ll - shape (seq_len, batch, k*input_size)
        x - shape (batch, input_size)
        :returns: logits
        """
        _, h = self.lc(ll)
        h = torch.squeeze(h, 0)
        s = F.relu(self.xs(x))
        d = F.relu(self.cd(h) + self.sd(s))
        y = self.dy(d)
        return y


def test_network(net, T: int, k: int, n_testing_samples: int, split: DataSplit):
    """Returns accuracy of the network on batch T"""
    net.eval()
    n_correct = 0
    n_total = 0
    for t in range(*split.batch_ind[T]):
        x = torch.from_numpy(split.features[t]).type(torch.FloatTensor).unsqueeze(0)
        y_hat = torch.zeros(N_ODOR_CLASSES).unsqueeze(0)
        for n in range(n_testing_samples):
            context = [choose_k(k, tau, split) for tau in range(T)]
            ll = torch.from_numpy(np.stack(context)).type(torch.FloatTensor).unsqueeze(1)
            y_hat += net(ll, x) / float(n_testing_samples)
        y = split.labels[t]
        if y == torch.argmax(y_hat.squeeze()).item():
            n_correct += 1
        n_total += 1
    net.train()
    return float(n_correct) / n_total


def train_epoch(net: nn.Module, criterion, optimizer, T: int, k: int, split: DataSplit) -> float:
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
    # allow target samples in batches 1...T-1
    sample_range_start = split.batch_ind[1][0]
    sample_range_end = split.batch_ind[T-1][1]

    # sample_indices[t] is the index of the sample to be processed at step t
    sample_indices = np.arange(sample_range_start, sample_range_end)
    np.random.shuffle(sample_indices)

    # sample_to_batch[s] is the batch number of sample index s
    sample_to_batch = np.empty(split.batch_ind[T-1][1], dtype=np.int)
    for batch, (start, end) in enumerate(split.batch_ind[:T]):
        sample_to_batch[start:end] = batch

    # train every sample in the epoch
    running_train_loss = 0.0
    for s in sample_indices:

        # if sample is in batch B, then sample context from batches up to B
        context_end_batch = sample_to_batch[s] # batch B >= 1; T > B
        assert(context_end_batch >= 1)
        assert(T > context_end_batch)
        context_length = 1 + np.random.choice(context_end_batch, 1).item() # n >= 1
        assert(context_length >= 1)
        context_start_batch = context_end_batch - context_length # B - n >= 0
        assert(context_start_batch >= 0)

        context = [choose_k(k, tau, split) for tau in range(context_start_batch, context_end_batch)]
        ll = torch.from_numpy(np.stack(context)).type(torch.FloatTensor).unsqueeze(1)
        x = torch.from_numpy(split.features[s]).type(torch.FloatTensor).unsqueeze(0)
        y = torch.tensor(split.labels[s], dtype=torch.long).unsqueeze(0)

        optimizer.zero_grad()
        y_pred = net(ll, x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    return running_train_loss / sample_indices.shape[0]


def calculate_stopping_time(patience: int, T: int, k: int):
    """Early stopping implementation.
    Train the network, evaluating its accuracy on the validation set at
    fixed intervals. If its accuracy dropped consecutively for a number of
    times equal to the patience parameter, then terminate training.

    Arguments:
        patience: the number of times the algorithm will tolerate a worse
            accuracy on the validation set before terminating
        T: the testing batch (model will be trained on batches 0...T-1
        k: the number of samples per class to use in context
    Saved outputs:
        val_lacc_{T}.pt: a list of sampled accuracies on the validation set
        val_loss_{T}.pt: a list of sampled losses on the train set
        val_li_{T}.pt: the training sample at which the above were evaluated
        val_stop_time_{T}.pt: a single integer, the stop time"""

    net = ContextModel(k)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(net)

    lloss = []
    lacc = []
    li = []
    consecutive_decrease_count = 0 #  number consecutive evaluations with decreasing accuracy

    validation_accuracy_prev = float('-inf')

    stopping_epoch = 0
    epoch = 0

    while consecutive_decrease_count < patience and epoch < 1000:
        # Adjust the model for one whole epoch
        avg_train_loss = train_epoch(net, criterion, optimizer, T, k, split_subtrain)
        scheduler.step()

        # Evaluate the network on the validation set
        validation_accuracy = test_network(net, T, k, 5, split_validate)

        # Update the patience parameter
        if validation_accuracy > validation_accuracy_prev:
            consecutive_decrease_count = 0
            validation_accuracy_prev = validation_accuracy
            stopping_epoch = epoch
        else:
            consecutive_decrease_count += 1

        # Add all the information to the logs
        lloss.append(avg_train_loss)
        lacc.append(validation_accuracy)
        li.append(epoch)

        epoch += 1

    torch.save(lloss, os.path.join(data_folder, f"val_lloss_{T}.pt"))
    torch.save(lacc, os.path.join(data_folder, f"val_lacc_{T}.pt"))
    torch.save(li, os.path.join(data_folder, f"val_li_{T}.pt"))
    torch.save(stopping_epoch, os.path.join(data_folder, f"val_stop_time_{T}.pt"))

    return stopping_epoch


def train_context_network(n_epochs: int, T: int, k: int):
    net = ContextModel(k)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(net)

    running_loss = 0.0
    lloss = []
    lacc = []
    li = []
    for epoch in range(n_epochs):
        # Adjust the model for one whole epoch
        avg_train_loss = train_epoch(net, criterion, optimizer, T, k, split_all)
        scheduler.step()

        # Evaluate the network on the validation set
        accuracy = test_network(net, T, k, 5, split_all)

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


def evaluate_network(n_testing_samples: int, T: int, k: int):
    """Load and evaluate the target network."""
    net = ContextModel(k)
    net.load_state_dict(torch.load(os.path.join(data_folder, f"model_{T}.pt")))
    acc = test_network(net, T, k, n_testing_samples, split_all)
    torch.save(acc, os.path.join(data_folder, f"acc_{T}.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    helpstr = "Specify the trial number to save different trial results to different output directories."
    parser.add_argument('-n', '--step', type=int, default=0, help=helpstr)
    args = parser.parse_args()
    step = args.step

    T, trial = step % 10, step // 10
    if T < 2:
        sys.exit()

    data_folder = f"output/context_short_harddecay_relu{trial}"
    try:
        os.makedirs(data_folder)
    except FileExistsError:
        pass

    # stopping_time = calculate_stopping_time(patience=5, T=T, k=5)
    train_context_network(n_epochs=MAX_EPOCHS, T=T, k=1)
    evaluate_network(n_testing_samples=100, T=T, k=1)
