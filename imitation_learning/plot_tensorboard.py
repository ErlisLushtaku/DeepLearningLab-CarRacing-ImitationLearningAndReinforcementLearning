import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def plot_tensorboard(tensorboard_dir):
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []
    steps = []

    for subdir, _, files in os.walk(tensorboard_dir):
        for file in files:
            if file.startswith("events.out.tfevents."):
                event_acc = EventAccumulator(os.path.join(subdir, file))
                event_acc.Reload()
                for tag in event_acc.Tags()['scalars']:
                    events = event_acc.Scalars(tag)
                    if 'loss_train' in tag:
                        train_losses.extend([event.value for event in events])
                        steps.extend([event.step for event in events])
                    elif 'loss_valid' in tag:
                        valid_losses.extend([event.value for event in events])
                    elif 'train_accuracy' in tag:
                        train_accuracy.extend([event.value for event in events])
                    elif 'validation_accuracy' in tag:
                        valid_accuracy.extend([event.value for event in events])

    # Find the maximum length among all data arrays
    max_len = max(len(train_losses), len(valid_losses), len(train_accuracy), len(valid_accuracy))

    # Resample the data arrays to match the maximum length
    train_losses = resample_data(train_losses, max_len)
    valid_losses = resample_data(valid_losses, max_len)
    train_accuracy = resample_data(train_accuracy, max_len)
    valid_accuracy = resample_data(valid_accuracy, max_len)

    # Plot the learning curves for loss
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(steps, train_losses, label='Training Loss')
    plt.plot(steps, valid_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss Learning Curves')
    plt.legend()
    plt.grid(True)

    # Plot the learning curves for accuracy
    plt.subplot(2, 1, 2)
    plt.plot(steps, train_accuracy, label='Training Accuracy')
    plt.plot(steps, valid_accuracy, label='Validation Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Learning Curves')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def resample_data(data, target_length):
    """
    Resamples the data array to match the target length.

    Args:
    - data (list): The input data array.
    - target_length (int): The target length to match.

    Returns:
    - list: The resampled data array.
    """
    if len(data) < target_length:
        # Oversample the data if it's shorter than the target length
        while len(data) < target_length:
            data.extend(data[:target_length - len(data)])
    elif len(data) > target_length:
        # Undersample the data if it's longer than the target length
        data = data[:target_length]

    return data

# Path to the directory containing TensorBoard event files
tensorboard_dir = "./tensorboard"
plot_tensorboard(tensorboard_dir)
