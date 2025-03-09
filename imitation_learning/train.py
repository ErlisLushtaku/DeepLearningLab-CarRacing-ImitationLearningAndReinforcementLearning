import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

import sys

sys.path.append(".")

import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    X_train_gray = np.array([utils.rgb2gray(img) for img in X_train])
    X_valid_gray = np.array([utils.rgb2gray(img) for img in X_valid])
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.
    y_train_discrete = np.array([utils.action_to_id(action) for action in y_train])
    y_valid_discrete = np.array([utils.action_to_id(action) for action in y_valid])

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    return X_train_gray, y_train_discrete, X_valid_gray, y_valid_discrete


def train_model(
    X_train,
    y_train,
    X_valid,
    n_minibatches,
    batch_size,
    lr,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent()

    tensorboard_eval = Evaluation(tensorboard_dir, "Imitation Learning", stats=["loss_train", "loss_valid", "train_accuracy", "validation_accuracy"])

    # implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    #
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     if i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)

    for i in range(n_minibatches):
        # Sample minibatch
        X_batch, y_batch = sample_minibatch(X_train, y_train, batch_size)

        # Update agent
        loss, train_accuracy = agent.update(X_batch, y_batch)

        if i % 10 == 0:
            # Compute training/ validation accuracy and write it to tensorboard
            train_loss = loss
            valid_loss, valid_accuracy = agent.compute_loss(X_valid, y_valid)
            tensorboard_eval.write_episode_data(
                i, {"loss_train": train_loss, "loss_valid": valid_loss, "train_accuracy": train_accuracy, "validation_accuracy": valid_accuracy}
            )

    # save your agent
    agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)


def sample_minibatch(X, y, batch_size):
    unique_classes = np.unique(y)
    samples_per_class = batch_size // len(unique_classes)

    X_batch = []
    y_batch = []

    for class_label in unique_classes:
        # Select indices corresponding to the current class
        class_indices = np.where(y == class_label)[0]

        # Randomly sample from the class indices
        selected_indices = np.random.choice(class_indices, size=samples_per_class, replace=False)

        # Add the selected samples to the minibatch
        X_batch.extend(X[selected_indices])
        y_batch.extend(y[selected_indices])

    # Convert to numpy arrays
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)

    return X_batch, y_batch


# def balance_dataset(X_train, y_train):
#     unique_actions, counts = np.unique(y_train, return_counts=True)
#     min_samples = np.min(counts)
#
#     balanced_X_train = []
#     balanced_y_train = []
#
#     for action in unique_actions:
#         indices = np.where(y_train == action)[0]
#
#         # Randomly sample without replacement to match the size of the smallest class
#         selected_indices = np.random.choice(indices, size=min_samples, replace=False)
#
#         balanced_X_train.extend(X_train[selected_indices])
#         balanced_y_train.extend(y_train[selected_indices])
#
#     return np.array(balanced_X_train), np.array(balanced_y_train)


def balance_dataset(X_train, y_train):
    unique_actions, counts = np.unique(y_train, return_counts=True)
    max_samples = np.max(counts)

    balanced_X_train = []
    balanced_y_train = []

    for action in unique_actions:
        indices = np.where(y_train == action)[0]
        num_samples = len(indices)

        if num_samples < max_samples:
            # Randomly sample with replacement to match max_samples
            selected_indices = np.random.choice(indices, size=max_samples, replace=True)
        else:
            selected_indices = indices

        balanced_X_train.extend(X_train[selected_indices])
        balanced_y_train.extend(y_train[selected_indices])

    return np.array(balanced_X_train), np.array(balanced_y_train)


if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=1
    )

    # balanced_X_train, balanced_y_train = balance_dataset(X_train, y_train)
    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=1500, batch_size=64, lr=1e-4)
