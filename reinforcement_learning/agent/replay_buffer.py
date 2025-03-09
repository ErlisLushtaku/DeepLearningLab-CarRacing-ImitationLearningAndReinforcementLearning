from collections import deque
import numpy as np
import os
import gzip
import pickle


class ReplayBuffer:

    def __init__(self, capacity=100000):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum capacity of the replay buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add_transition(self, state, action, next_state, reward, done):
        """
        Add a transition to the replay buffer.

        Args:
            state: Current state.
            action: Action taken.
            next_state: Next state.
            reward: Reward received.
            done: Whether the episode terminated after this transition.
        """
        self.buffer.append((state, action, next_state, reward, done))

    def next_batch(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple: Batch of sampled transitions.
        """

        batch_indices = np.random.choice(len(self.buffer), batch_size)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = zip(
            *[self.buffer[idx] for idx in batch_indices]
        )
        return (
            np.array(batch_states),
            np.array(batch_actions),
            np.array(batch_next_states),
            np.array(batch_rewards),
            np.array(batch_dones),
        )
