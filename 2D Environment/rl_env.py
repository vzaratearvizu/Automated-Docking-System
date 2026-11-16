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
EPISODE_TIME_LIMIT_SECONDS = 50
# How many frames until the episode ends
EPISODE_TIME_LIMIT_FRAMES = EPISODE_TIME_LIMIT_SECONDS * FPS
TIMEOUT_PENALTY = -5
SUCCESS_BONUS = 10
MAX_SPEED = 3.0

class MooringEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.boat = Boat(0, 0, 40, 100)
        self.env = environments.mooring_field

        # Creating a lower and higher bound for possible actions to take.
        self.action_space = spaces.Box(
            low=numpy.array([-1.0, -1.0], dtype=numpy.float32),
            high=numpy.array([1.0, 1.0], dtype=numpy.float32),
        )
        # Creating a lower and higher bound for possible observations as well as the shape of seeing the observations
        self.observation_space = spaces.Box(
            low=-numpy.inf, high=numpy.inf, shape=(6,), dtype=numpy.float32
        )

        # Keeping track of how many steps have passed for the episde time limit
        self.steps_elapsed = 0
        self.starting_distance = 1
        self.best_distance = 1

    def build_observation(self):
        base_observation = self.env.get_observation(self.boat)
        best_progress = self.best_distance / self.starting_distance
        return numpy.append(base_observation, numpy.float32(best_progress)).astype(numpy.float32)

    # Resetting the environment whether success or fail. The asterisk is just convention so when things outside my local
    # system call it, they require the positional arguments to be labeled by name, not position. seed and options are here just because because
    # gymnasium method signature requires so, it's not being used at all.
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.boat.reset(random.uniform(300, 870), random.uniform(300, 700), random.uniform(-math.pi/2, math.pi/2))
        self.env.reset()
        self.steps_elapsed = 0
        self.starting_distance = self.env.distance_to_buoy(self.boat)
        self.best_distance = self.starting_distance
        return self.build_observation(), {}

    # Makes a decision that affects the environment
    def step(self, action):
        throttle, steering = float(action[0]), float(action[1])
        self.boat.update(throttle, steering)
        self.steps_elapsed += 1

        current_distance = self.env.distance_to_buoy(self.boat)

        progress = max(0, self.best_distance - current_distance) / self.starting_distance
        self.best_distance = min(self.best_distance, current_distance)
        reward = progress

        # Sees if we're within the radius of the buoy
        outcome = self.env.check_outcome(self.boat)
        timed_out = self.steps_elapsed >= EPISODE_TIME_LIMIT_FRAMES
        terminated = False
        truncated = False

        if outcome == "docked":
            reward += SUCCESS_BONUS
            terminated = True
        elif timed_out:
            reward += TIMEOUT_PENALTY
            truncated = True
            outcome = "timeout"

        observation = self.build_observation()
        # info dictionary is just needed as a gymnasium signature for providing more info on what's happening
        info = {"outcome": outcome}
        # Returning what's below because after an action is made, we need to see what happened, the reward, whether the episode
        # ended due to success, whether it ended to due a taking too long, then the extra info on what's happening. This is fed
        # and used in the train.py file. The observation turns into next_observation, reward is added to the episode_reward,
        # terminated becomes done, trauncated determines if env.reset is called, and info is just printed in the per-epsidoee
        # status line.
        return observation, reward, terminated, truncated, info