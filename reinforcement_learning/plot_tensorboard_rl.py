import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def plot_tensorboard(tensorboard_dir):
    """
    Plot episode rewards from TensorBoard logs.
    Shows both training and evaluation rewards with different colors.
    """
    # Dictionary to store data: {tag_name: {'steps': [], 'rewards': []}}
    all_data = {}

    for subdir, _, files in os.walk(tensorboard_dir):
        for file in files:
            if file.startswith("events.out.tfevents."):
                filepath = os.path.join(subdir, file)
                event_acc = EventAccumulator(filepath)
                event_acc.Reload()
                
                for tag in event_acc.Tags()['scalars']:
                    # Only plot reward-related tags
                    if 'episode_reward' in tag:
                        events = event_acc.Scalars(tag)
                        steps = [event.step for event in events]
                        rewards = [event.value for event in events]
                        
                        if tag not in all_data:
                            all_data[tag] = {'steps': steps, 'rewards': rewards}
                        else:
                            all_data[tag]['steps'].extend(steps)
                            all_data[tag]['rewards'].extend(rewards)

    if not all_data:
        print("No episode_reward data found.")
        return

    plt.figure(figsize=(12, 6))
    
    # Define colors and styles for different tags
    styles = {
        'episode_reward': {'color': 'blue', 'alpha': 0.5, 'label': 'Training Episode Reward'},
        'eval_episode_reward': {'color': 'red', 'alpha': 0.9, 'label': 'Evaluation Episode Reward', 'linewidth': 2},
    }
    
    for tag, data in all_data.items():
        # Sort by steps to ensure proper line drawing
        sorted_pairs = sorted(zip(data['steps'], data['rewards']))
        steps, rewards = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        style = styles.get(tag, {'color': 'green', 'alpha': 0.7, 'label': tag})
        plt.plot(steps, rewards, 
                 color=style.get('color', 'green'),
                 alpha=style.get('alpha', 0.7),
                 label=style.get('label', tag),
                 linewidth=style.get('linewidth', 1))
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('DQN Learning Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Path to the directory containing TensorBoard event files
tensorboard_dir = "./tensorboard"
plot_tensorboard(tensorboard_dir)
