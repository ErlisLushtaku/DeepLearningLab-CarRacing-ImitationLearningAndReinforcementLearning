import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
        self,
        Q,
        Q_target,
        num_actions,
        gamma=0.95,
        batch_size=64,
        epsilon=0.1,
        tau=0.01,
        lr=1e-4,
        history_length=0,
        replay_capacity=100000,
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float between 0 and 1.
           lr: learning rate of the optimizer
           history_length: Length of the history to be concatenated as input.
           replay_capacity: Maximum capacity of the replay buffer.
        """
        # setup networks
        self.Q = Q
        self.Q_target = Q_target
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # Add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        # Sample next batch
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)

        # Convert batch to tensors
        states_tensor = torch.tensor(batch_states, dtype=torch.float32)
        actions_tensor = torch.tensor(batch_actions, dtype=torch.int64).unsqueeze(1)
        next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32)
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1)
        dones_tensor = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1)

        # Compute TD targets
        with torch.no_grad():
            Q_next = self.Q_target(next_states_tensor)
            Q_next_max, _ = torch.max(Q_next, dim=1, keepdim=True)
            td_target = rewards_tensor + self.gamma * Q_next_max * (1 - dones_tensor)

        # Update the Q network
        self.optimizer.zero_grad()
        Q_pred = self.Q(states_tensor)
        Q_pred_action = Q_pred.gather(1, actions_tensor)
        loss = self.loss_function(Q_pred_action, td_target)
        loss.backward()
        self.optimizer.step()

        # Call soft update for target network
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # take greedy action (argmax)
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                Q_values = self.Q(state_tensor)
                action_id = torch.argmax(Q_values).item()
        else:
            # sample random action
            STRAIGHT = 0
            LEFT = 1
            RIGHT = 2
            ACCELERATE = 3
            BRAKE = 4

            action_probs = [0.3, 0.15, 0.15, 0.3, 0.1]
            action_id = np.random.choice(self.num_actions, p=action_probs)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
