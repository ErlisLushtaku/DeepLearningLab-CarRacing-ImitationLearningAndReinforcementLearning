from __future__ import print_function

import os
import json
from datetime import datetime
import gym
from agent.dqn_agent import DQNAgent
from imitation_learning.agent.networks import CNN
from train_carracing import run_episode
from agent.networks import MLP
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    # Define networks and load agent
    Q_network = CNN()
    Q_target_network = CNN()

    agent = DQNAgent(
        Q=Q_network,
        Q_target=Q_target_network,
        num_actions=5,
        gamma=0.99,
        batch_size=64,
        epsilon=0.2,
        tau=0.01,
        lr=1e-4,
        history_length=0,  # assuming no history
    )

    # Load trained agent weights
    agent.load("./models_carracing/dqn_agent.pt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent, deterministic=True, do_training=False, rendering=True, max_timesteps=1500,
        )
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    with open(fname, "w") as fh:
        json.dump(results, fh)

    env.close()
    print("... finished")
