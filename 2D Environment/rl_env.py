# Gymnasium wrapper tying boat.py + environments.py together for training.
# Separate from main.py's manual-test loop, this is what train.py will use.

import math
import numpy
import gymnasium as gym
from gymnasium import spaces
import random

from boat import Boat
import environments

# 60 frames a second
FPS = 60
# How many seconds until the episode ends
EPISODE_TIME_LIMIT_SECONDS = 60
# How many frames until the episode ends
EPISODE_TIME_LIMIT_FRAMES = EPISODE_TIME_LIMIT_SECONDS * FPS
MAX_SPEED = 3.0
TIMEOUT_PENALTY = -100.0

class MooringEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.boat = Boat(random.uniform(50, 950), random.uniform(300, 700), 40, 100)
        self.env = environments.mooring_field

        # Creating a lower and higher bound for possible actions to take.
        self.action_space = spaces.Box(
            low=numpy.array([-1.0, -1.0], dtype=numpy.float32),
            high=numpy.array([1.0, 1.0], dtype=numpy.float32),
        )
        # Creating a lower and higher bound for possible observations as well as the shape of seeing the observations
        self.observation_space = spaces.Box(
            low=-numpy.inf, high=numpy.inf, shape=(5,), dtype=numpy.float32
        )

        # Keeping track of how many steps have passed for the episde time limit
        self.steps_elapsed = 0

    # Resetting the environment whether success or fail. The asterisk is just convention so when things outside my local
    # system call it, they require the positional arguments to be labeled by name, not position. seed and options are here just because because
    # gymnasium method signature requires so, it's not being used at all.
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.boat.reset(random.uniform(50, 950), random.uniform(300, 700))
        self.env.reset()
        self.steps_elapsed = 0
        observation = self.env.get_observation(self.boat)
        # Things we need for making decisions:
            # the relative angle the boat is facing [0]
            # the angle difference from the boat's heading to the angle where the buoy is [1]
            # the boats speed [2]
            # the velocity of the boat's turning [3]
            # the distance between the boat and buoy [4]
        self.starting_distance = observation[4]
        # Return the observation and nothing because that's how gymasium requires reset to return something, observation and info. Nothing extra 
        # to report at the start of an episode, so an empty dictis a complete, valid answer here, not a placeholder for something missing.
        return observation, {}

    # Makes a decision that affects the environment
    def step(self, action):
        throttle, steering = float(action[0]), float(action[1])
        self.boat.update(throttle, steering)

        self.steps_elapsed += 1

        # Sees if we're within the radius of the buoy
        outcome = self.env.check_outcome(self.boat)
        timed_out = self.steps_elapsed >= EPISODE_TIME_LIMIT_FRAMES
        observation = self.env.get_observation(self.boat)
        # Things we need for making decisions:
            # the relative angle the boat is facing [0]
            # the angle difference from the boat's heading to the angle where the buoy is [1]
            # the boats speed [2]
            # the velocity of the boat's turning [3]
            # the distance between the boat and buoy [4]
        current_distance = observation[4]
        terminated = False
        truncated = False

        if outcome == "MOOOORRRREEEEDDD":
            # Around 1.0 around success
            reward_multiplier = self.env.zone_progress(self.boat)
            reward = reward_multiplier * 25
            terminated = True
            outcome = "docked"
        elif timed_out:
            reward = TIMEOUT_PENALTY
            truncated = True
            outcome = "timeout"
        else:
            distance_ratio = current_distance / self.starting_distance
            reward_multiplier = self.env.zone_progress(self.boat)
            # 360 frames to be awarded well, 6 seconds at 60 frames. So if it started already in the radius and 
            # stayed in, the math would be 
            # increments of 60/360 * 20 is about 35,900
            reward = -distance_ratio + reward_multiplier * 25

        # info dictionary is just needed as a gymnasium signature for providing more info on what's happening
        info = {"outcome": outcome}
        # Returning what's below because after an action is made, we need to see what happened, the reward, whether the episode
        # ended due to success, whether it ended to due a taking too long, then the extra info on what's happening. This is fed
        # and used in the train.py file. The observation turns into next_observation, reward is added to the episode_reward,
        # terminated becomes done, trauncated determines if env.reset is called, and info is just printed in the per-epsidoee
        # status line.
        return observation, reward, terminated, truncated, info