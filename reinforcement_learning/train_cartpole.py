import sys
import os

sys.path.append("../")

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats


def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    model_dir="./models_cartpole",
    tensorboard_dir="./tensorboard/cartpole",
    eval_cycle=20,
    num_eval_episodes=5,
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir), "train", ["episode_reward", "a_0", "a_1", "mean_greedy_reward"])

    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "a_0": stats.get_action_usage(0),
                "a_1": stats.get_action_usage(1),
            },
        )

        # Evaluate agent every 'eval_cycle' episodes using greedy actions only
        if i % eval_cycle == 0:
            eval_rewards = []
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False)
                eval_rewards.append(eval_stats.episode_reward)
            mean_greedy_reward = np.mean(eval_rewards)
            tensorboard.write_episode_data(i, eval_dict={"mean_greedy_reward": mean_greedy_reward})

        # Store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


if __name__ == "__main__":
    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 20  # evaluate every 20 episodes

    # CartPole environment
    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # Initialize Q network and target network
    Q_network = MLP(state_dim, num_actions)
    Q_target_network = MLP(state_dim, num_actions)

    # Initialize DQN agent
    agent = DQNAgent(
        Q=Q_network,
        Q_target=Q_target_network,
        num_actions=num_actions,
        gamma=0.99,
        batch_size=64,
        epsilon=0.2,
        tau=0.01,
        lr=1e-4,
        history_length=0,  # assuming no history
    )

    # Train DQN agent
    train_online(
        env, agent, num_episodes=1000, eval_cycle=eval_cycle, num_eval_episodes=num_eval_episodes
    )
