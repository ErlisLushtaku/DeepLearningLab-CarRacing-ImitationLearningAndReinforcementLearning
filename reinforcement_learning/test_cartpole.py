import os
from datetime import datetime
import gym
import json
from agent.dqn_agent import DQNAgent
from train_cartpole import run_episode
from agent.networks import MLP
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v0").unwrapped

    # Load DQN agent
    Q_network = MLP(state_dim=4, action_dim=2)  # Assuming state_dim and action_dim are correct
    Q_target_network = MLP(state_dim=4, action_dim=2)  # Assuming state_dim and action_dim are correct
    num_actions = 2  # Assuming 2 actions for CartPole environment
    agent = DQNAgent(Q_network, Q_target_network, num_actions)
    agent.load("./models_cartpole/dqn_agent.pt")  # Assuming the path to the saved agent model is correct

    n_test_episodes = 5

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(
            env, agent, deterministic=True, do_training=False, rendering=True
        )
        episode_rewards.append(stats.episode_reward)

    # Save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    with open(fname, "w") as fh:
        json.dump(results, fh)

    env.close()
    print("... finished")
