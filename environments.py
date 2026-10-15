# Holds the different environments the boat will dock in
# Choose which one to use in main.py

import math
import pygame

REQUIRED_SECONDS = 6
# This has got to match the clock.tick(ANY NUM) in main.py
FPS = 60
REQUIRED_FRAMES = REQUIRED_SECONDS * FPS

class MooringField:
    def __init__(self):
        self.buoy_point = (100, 50)
        # How close the center boat needs to be
        self.catch_radius = 70
        self.frames_in_zone = 0

    def draw_env(self, surface):
        # The buoy itself
        pygame.draw.circle(surface, (220, 60, 60), self.buoy_point, 10)
        pygame.draw.circle(surface, (220, 60, 60), self.buoy_point, self.catch_radius, width=1)

    def check_outcome(self, boat):
        """Returns 'moored' if the boat's center is within catch_radius of
        the buoy, regardless of speed or heading. Returns None otherwise."""
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

# Going ahead and initializing the environments so we can just 
#     import environments
#     current_env = environments.whatever
mooring_field = MooringField()