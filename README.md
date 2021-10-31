# A context-based gas sensor classifier

Please see the paper here: https://arxiv.org/abs/2003.07292

The context-based gas classifier processes a sequence of labeled odor samples in order to form a representation of the current sensor context, containing information pertaining to sensor drift. This context vector is used to inform the prediction of the odor label.

## Usage

This repository includes three useful files:

* The main file that starts the training of the network with the GA (**main_odor_GA.py**)
* The neural network module that contains the class for the modified Context-Odor Model (mCOM, **neural_net.py**)
* The module that reads the batches of data (**read_batches.py**)

## Quick set-up for the GA run

data_available = **False** by default. If the training data was loaded before, you can change it to **True**.

trainingTASKs = [1, 2, 3, 4, 5]. This is the list of tasks (batches) that the mCOM will be trained on. It has to be always more than 1 due to the Context module. Context module's, i.e., LSTM cell, memory is reset after each batch. In order to compare mCOM with the Baseline Model, Line-78 (net.resetContextMemory()) can be commented out.
