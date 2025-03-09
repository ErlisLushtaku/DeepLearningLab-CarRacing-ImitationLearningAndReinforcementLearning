import math
import sys
import os
import numpy as np
import gym

from imitation_learning.agent.networks import CNN
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray, id_to_action, ActionParams
from agent.dqn_agent import DQNAgent

np.random.seed(0)

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


def run_episode(
    env,
    agent,
    deterministic,
    skip_frames=3,
    do_training=True,
    rendering=True,
    max_timesteps=1000,
    history_length=0,
    action_params=None
):
    """
    This method runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according to the Q function approximator (no random actions).
    do_training == True => train agent
    """
    params = ActionParams()
    if action_params is not None:
        for key, value in action_params.__dict__.items():
            setattr(params, key, value)

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(history_length + 1, 96, 96)

    while True:

        # Get action_id from agent
        action_id = agent.act(state, deterministic=deterministic)
        action = id_to_action(action_id)

        # env.car.hull.linearVelocity[0] = np.clip(env.car.hull.linearVelocity[0], -params.maxSpeed, params.maxSpeed)
        # env.car.hull.linearVelocity[1] = np.clip(env.car.hull.linearVelocity[1], -params.maxSpeed, params.maxSpeed)
        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(history_length + 1, 96, 96)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(
    env,
    agent,
    num_episodes,
    history_length=0,
    model_dir="./models_carracing",
    tensorboard_dir="./tensorboard/carracing",
):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(
        os.path.join(tensorboard_dir),
        "train",
        ["episode_reward", "straight", "left", "right", "accel", "brake", "eval_episode_reward"],
    )

    for i in range(num_episodes):
        print("epsiode %d" % i)

        max_timesteps = min(math.exp(0.069/2 * i) + 100, 800)
        print("timesteps ", max_timesteps)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            do_training=True,
        )

        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "straight": stats.get_action_usage(STRAIGHT),
                "left": stats.get_action_usage(LEFT),
                "right": stats.get_action_usage(RIGHT),
                "accel": stats.get_action_usage(ACCELERATE),
                "brake": stats.get_action_usage(BRAKE),
            },
        )

        # evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        if i % eval_cycle == 0:
            eval_stats = run_episode(
                env,
                agent,
                deterministic=True,
                do_training=False,
            )

            tensorboard.write_episode_data(
                i,
                eval_dict={
                    "eval_episode_reward": eval_stats.episode_reward,
                },
            )

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20

    env = gym.make("CarRacing-v0").unwrapped

    # Define Q network, target network, and DQN agent
    Q_network = CNN()  # Assuming CNN is your CNN architecture
    Q_target_network = CNN()  # Assuming CNN is your CNN architecture
    num_actions = 5  # Assuming 5 actions for CarRacing environment
    agent = DQNAgent(
        Q_network,
        Q_target_network,
        num_actions,
        history_length=0,  # Adjust if you are using a history of frames
    )

    train_online(
        env,
        agent,
        num_episodes=1000,
        history_length=0,  # Adjust if you are using a history of frames
        model_dir="./models_carracing",
        tensorboard_dir="./tensorboard",
    )
