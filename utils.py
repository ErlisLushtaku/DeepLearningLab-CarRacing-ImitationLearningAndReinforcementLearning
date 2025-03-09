import math

import numpy as np

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    return gray.astype("float32")


def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT  # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT  # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE  # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]):
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0


class ActionParams:
    def __init__(self,
                 maxSpeed=77.0,
                 exponentialIncreaseOfSpeedParameter=0.0157,
                 logarithmicDecreaseOfSpeedParameter=3.0,
                 highSpeedDecreaseParameter=0.35,
                 brakingDuringTurningSpeedThresholdParameter=45.0,
                 brakingDuringTurningParameter=0.3,
                 brakingParameter=0.6,
                 accelerationDuringTurningParameter=0.5,
                 accelerationParameter=5.0,
                 straightParameter=0.5):
        self.maxSpeed = maxSpeed
        self.exponentialIncreaseOfSpeedParameter = exponentialIncreaseOfSpeedParameter
        self.logarithmicDecreaseOfSpeedParameter = logarithmicDecreaseOfSpeedParameter
        self.highSpeedDecreaseParameter = highSpeedDecreaseParameter
        self.brakingDuringTurningSpeedThresholdParameter = brakingDuringTurningSpeedThresholdParameter
        self.brakingDuringTurningParameter = brakingDuringTurningParameter
        self.accelerationDuringTurningParameter = accelerationDuringTurningParameter
        self.accelerationParameter = accelerationParameter
        self.brakingParameter = brakingParameter
        self.straightParameter = straightParameter

def modified_id_to_action(action_id, env, params=None):
    """
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])
    trueSpeed = np.sqrt(
        np.square(env.car.hull.linearVelocity[0])
        + np.square(env.car.hull.linearVelocity[1])
    )

    argument = -trueSpeed + params.maxSpeed + 1
    if argument <= 0:
        argument = 1

    logarithmicDecreaseOfSpeed = math.log(argument / params.logarithmicDecreaseOfSpeedParameter) / math.log((params.maxSpeed + 1) / params.logarithmicDecreaseOfSpeedParameter)
    exponentialIncreaseOfSpeed = math.exp(trueSpeed * params.exponentialIncreaseOfSpeedParameter) - 1.0

    # logarithmicIncreaseOfSpeed = min(math.log(trueSpeed + 1)/1.5 + 0.3, logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter)

    # turning = logarithmicDecreaseOfSpeed
    # if params.amountOrLogarithmicDecreaseOfSpeed == 'amount':
    #     turning = amount

    # if trueSpeed < 8:
    #     leftTurningAmount = -logarithmicIncreaseOfSpeed
    # el
    # if trueSpeed >= params.speedDecreaseThresholdParameter:
    #     leftTurningAmount = -logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter * 0.2
    # elif trueSpeed >= params.speedDecreaseThresholdParameter - 10:
    #     leftTurningAmount = -logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter * 0.6
    # elif trueSpeed >= params.speedDecreaseThresholdParameter - 20:
    #     leftTurningAmount = -logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter * 0.9
    # else:
    #     leftTurningAmount = -logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter

    # if trueSpeed < 8:
    #     rightTurningAmount = logarithmicIncreaseOfSpeed
    # el
    # if trueSpeed >= params.speedDecreaseThresholdParameter:
    #     rightTurningAmount = logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter * 0.2
    # elif trueSpeed >= params.speedDecreaseThresholdParameter - 10:
    #     rightTurningAmount = logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter * 0.6
    # elif trueSpeed >= params.speedDecreaseThresholdParameter - 20:
    #     rightTurningAmount = logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter * 0.9
    # else:
    #     rightTurningAmount = logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter

    # accelerationDuringTurning = 0.0
    # if trueSpeed < params.brakingDuringTurningSpeedThresholdParameter - 30:
    #     accelerationDuringTurning = logarithmicDecreaseOfSpeed * params.accelerationDuringTurningParameter
    # elif trueSpeed < params.brakingDuringTurningSpeedThresholdParameter - 20:
    #     accelerationDuringTurning = logarithmicDecreaseOfSpeed * params.accelerationDuringTurningParameter * 0.9
    # elif trueSpeed < params.brakingDuringTurningSpeedThresholdParameter - 10:
    #     accelerationDuringTurning = logarithmicDecreaseOfSpeed * params.accelerationDuringTurningParameter * 0.6
    # elif trueSpeed < params.brakingDuringTurningSpeedThresholdParameter:
    #     accelerationDuringTurning = logarithmicDecreaseOfSpeed * params.accelerationDuringTurningParameter * 0.2


    # brakingDuringTurning = 0.0
    # if trueSpeed >= params.brakingDuringTurningSpeedThresholdParameter:
    #     brakingDuringTurning = exponentialIncreaseOfSpeed * params.brakingDuringTurningParameter * 0.15
    # elif trueSpeed >= params.brakingDuringTurningSpeedThresholdParameter + 10:
    #     brakingDuringTurning = exponentialIncreaseOfSpeed * params.brakingDuringTurningParameter * 0.6
    # elif trueSpeed >= params.brakingDuringTurningSpeedThresholdParameter + 20:
    #     brakingDuringTurning = exponentialIncreaseOfSpeed * params.brakingDuringTurningParameter * 0.9
    # elif trueSpeed >= params.brakingDuringTurningSpeedThresholdParameter + 30:
    #     brakingDuringTurning = exponentialIncreaseOfSpeed * params.brakingDuringTurningParameter


    # acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter
    # if trueSpeed >= 10:
    #     acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter * 0.8
    # elif trueSpeed >= 20:
    #     acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter * 0.2
    # elif trueSpeed >= 30:
    #     acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter * 0.1
    # elif trueSpeed >= 40:
    #     acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter * 0.05
    # elif trueSpeed >= 50:
    #     acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter * 0.002
    # elif trueSpeed >= 60:
    #     acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter * 0.001
    # elif trueSpeed >= 70:
    #     acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter * 0.0005
    # elif trueSpeed >= 80:
    #     acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter * 0.0002

    leftTurningAmount = -logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter
    rightTurningAmount = logarithmicDecreaseOfSpeed * params.highSpeedDecreaseParameter

    brakingDuringTurning = exponentialIncreaseOfSpeed * params.brakingDuringTurningParameter if trueSpeed >= params.brakingDuringTurningSpeedThresholdParameter else 0.0
    braking = exponentialIncreaseOfSpeed * params.brakingParameter

    accelerationDuringTurning = logarithmicDecreaseOfSpeed * params.accelerationDuringTurningParameter if trueSpeed < params.brakingDuringTurningSpeedThresholdParameter else 0.0
    acceleration = logarithmicDecreaseOfSpeed * params.accelerationParameter
    nler = logarithmicDecreaseOfSpeed * params.straightParameter

    if action_id == LEFT:
        return np.array([leftTurningAmount,
                         accelerationDuringTurning,
                         brakingDuringTurning])
    elif action_id == RIGHT:
        return np.array([rightTurningAmount,
                         accelerationDuringTurning,
                         brakingDuringTurning])
    elif action_id == ACCELERATE:
        return np.array([0.0, acceleration, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, braking])
    else:
        return np.array([0.0, nler, 0.0])

def id_to_action(action_id, max_speed=0.5):
    """
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.05, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.05, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 1.0])
    else:
        return np.array([0.0, 0.2, 0.0])

class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)
