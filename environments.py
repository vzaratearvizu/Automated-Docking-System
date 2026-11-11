# Holds the different environments the boat will dock in
# Choose which one to use in main.py

import math
import numpy
import pygame
import random

# Amount of seconds needed to successfully park
REQUIRED_SECONDS = 6
# This has got to match the clock.tick(ANY NUM) in main.py
FPS = 60
REQUIRED_FRAMES = REQUIRED_SECONDS * FPS

class MooringField:
    def __init__(self):
        # The point on the screen where the buoy is
        self.buoy_point = (random.uniform(50, 940), random.uniform(60, 190))
        # How close to the buoy the center of the boat needs to be to register
        self.catch_radius = 70
        # Tracks how many frames the boat is in the radius
        self.frames_in_zone = 0

    def draw_env(self, surface):
        # The buoy itself
        pygame.draw.circle(surface, (220, 60, 60), self.buoy_point, 10)
        # The radius
        pygame.draw.circle(surface, (220, 60, 60), self.buoy_point, self.catch_radius, width=1)

    def check_outcome(self, boat):
        distance_x = boat.x_pos - self.buoy_point[0]
        distance_y = boat.y_pos - self.buoy_point[1]
        distance = math.hypot(distance_x, distance_y)

        if distance <= self.catch_radius:
            self.frames_in_zone += 1
        else:
            self.frames_in_zone = 0

        if self.frames_in_zone >= REQUIRED_FRAMES:
            return "MOOOORRRREEEEDDD"
        return None

    def reset(self):
        self.frames_in_zone = 0
        self.buoy_point = (random.uniform(50, 940), random.uniform(60, 190))

    def get_observation(self, boat):
        distance_x = self.buoy_point[0] - boat.x_pos
        distance_y = self.buoy_point[1] - boat.y_pos
        distance = math.hypot(distance_x, distance_y)

        # Negative with distance_y since the positive side actually goes down instead of up like in a standard math grid.
        # Bearing is the direction of a point from the vessel (boat) to the object
        # We are measuring the current angle, more accurately creating a reference line by making a line from the center of the vessels distance to the buoy
        bearing_object_reference = math.atan2(distance_x, -distance_y)

        # Figures out the angle distance between the heading (front) of the boat to the buoy
        angle_to_target = bearing_object_reference - boat.heading
        # angle_to_target = 179 - -179 = 358
        # angle_to_target = (358 + 180) mod 360 - 180
        # angle_to_target = 538 mod 360 - 180
        # angle_to_target = 178 remainder - 180 = -2 adjustment
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi

        # The boat's heading could be spinning multiple times beyond -+360. So this converts it back between -+180 degrees. 0 is facing upwards.
        heading_relative_angle = (boat.heading + math.pi) % (2 * math.pi) - math.pi

        # Things we need for making decisions:
        # the relative angle the boat is facing
        # the angle difference from the boat's heading to the angle where the buoy is
        # the boats speed
        # the velocity of the boat's turning
        # the distance between the boat and buoy
        return numpy.array(
            [heading_relative_angle, angle_to_target, boat.speed, boat.angular_velocity, distance], 
            dtype=numpy.float32)

# Going ahead and initializing the environments so we can just 
#     import environments
#     current_env = environments.whatever
mooring_field = MooringField()