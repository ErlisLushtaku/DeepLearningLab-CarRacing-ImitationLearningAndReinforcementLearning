import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def plot_tensorboard(tensorboard_dir):
    episode_reward = []
    steps = []

    for subdir, _, files in os.walk(tensorboard_dir):
        for file in files:
            if file.startswith("events.out.tfevents."):
                event_acc = EventAccumulator(os.path.join(subdir, file))
                event_acc.Reload()
                for tag in event_acc.Tags()['scalars']:
                    events = event_acc.Scalars(tag)
                    if 'episode_reward' in tag:
                        episode_reward.extend([event.value for event in events])
                        steps.extend([event.step for event in events])

    # Check if data is available before plotting
    if episode_reward:
        # Plot the learning curves for episode_reward
        plt.plot(steps, episode_reward, label='episode_reward')
        plt.xlabel('Steps')
        plt.ylabel('Episode Reward')
        plt.title('Episode Reward')
        plt.legend()
        plt.show()
    else:
        print("No episode_reward data found.")

# Path to the directory containing TensorBoard event files
tensorboard_dir = "./tensorboard"
plot_tensorboard(tensorboard_dir)
