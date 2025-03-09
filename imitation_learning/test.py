import itertools
import sys

sys.path.append(".")
from datetime import datetime
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=2000, action_params=None):
    episode_reward = 0
    step = 0

    params = ActionParams()
    if action_params is not None:
        for key, value in action_params.__dict__.items():
            setattr(params, key, value)

    state = env.reset()

    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()

    while True:

        # preprocess the state in the same way as in your preprocessing in train.py
        #    state = ...
        state = rgb2gray(state)

        # get the action from your agent! You need to transform the discretized actions to continuous actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...

        action_id, _ = agent.predict(state)
        a = modified_id_to_action(action_id, env, params)
        # a = id_to_action(action_id)
        # Clip acceleration if necessary
        # a[1] = np.clip(a[1], 0.0, 0.1)  # Clip acceleration to avoid going too fast
        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if params.maxSpeed <= 100:
            env.car.hull.linearVelocity[0] = np.clip(env.car.hull.linearVelocity[0], -params.maxSpeed, params.maxSpeed)
            env.car.hull.linearVelocity[1] = np.clip(env.car.hull.linearVelocity[1], -params.maxSpeed, params.maxSpeed)
        else:
            env.car.hull.linearVelocity[0] = env.car.hull.linearVelocity[0] * params.maxSpeed / 100
            env.car.hull.linearVelocity[1] = env.car.hull.linearVelocity[1] * params.maxSpeed / 100

        print()
        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":
    # Instantiate the BCAgent
    agent = BCAgent()

    # Load the trained model
    agent.load("models/agent.pt")

    # Important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15  # number of episodes to test


    # Define parameter ranges for tuning
    # param_ranges = {
    #     "maxSpeed": [30.0, 115.0, 5.0],
    # }
    #
    # # Generate values for each parameter
    # param_values = {param: np.arange(start, stop, step) for param, (start, stop, step) in param_ranges.items()}
    param_values = {
        # "maxSpeed": [77.0, 80.0, 84.0],
        "highSpeedDecreaseParameter": [1.5, 2.0, 2.5, 3.0],
        "logarithmicDecreaseOfSpeedParameter": [1.0, 2.0, 3.0, 4.0],
        "exponentialIncreaseOfSpeedParameter": [0.01, 0.011, 0.012, 0.013, 0.014],
        "brakingDuringTurningSpeedThresholdParameter": [40.0, 47.0, 55.0],
        "brakingDuringTurningParameter": [0.4, 0.5, 0.6, 0.75],
        # "brakingParameter": [0.5, 1.0, 3.0],
        # "accelerationDuringTurningParameter": [1.0, 2.0],
        # "accelerationParameter": [2.0, 2.7, 3.5],
        # "straightParameter": [0.0, 0.05, 0.5, 1.0, 2.0, 3.0],
    }

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(*param_values.values()))

    for params in param_combinations:
        try:
            # Code that may raise an exception
            env = gym.make("CarRacing-v0").unwrapped
            # Create an ActionParams object with the current combination of parameters
            action_params = ActionParams(**dict(zip(param_values.keys(), params)))
            results_list = []

            for i in range(n_test_episodes):
                episode_reward = run_episode(env, agent, rendering=rendering, action_params=None)
                results_list.append(episode_reward)

            # Calculate mean and standard deviation of episode rewards
            mean_reward = sum(results_list) / len(results_list)
            std_reward = (sum((r - mean_reward) ** 2 for r in results_list) / len(results_list)) ** 0.5

            # Save results to a JSON file
            directory = "results/auto"
            if not os.path.exists(directory):
                os.makedirs(directory)

            fname = os.path.join(directory, f"results_bc_agent-auto-3-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
            with open(fname, "w") as fh:
                json.dump({
                    "parameters": action_params.__dict__,
                    "episode_rewards": results_list,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward
                }, fh)

            env.close()
            break
        except Exception as e:
            # Handle other exceptions
            print("An error occurred:", e)

    print("... finished")
