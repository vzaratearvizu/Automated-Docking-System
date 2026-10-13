# Holds the different environments the boat will dock in
# Choose which one to use in main.py

import pygame

class MooringField:
    def __init__(self):
        self.buoy_point = (100, 50)
        # How close the center boat needs to be
        self.catch_radius = 20

    def draw_env(self, surface):
        # The buoy itself
        pygame.draw.circle(surface, (220, 60, 60), self.buoy_point, 10)
        pygame.draw.circle(surface, (220, 60, 60), self.buoy_point, self.catch_radius, width=1)

# Going ahead and initializing the environments so we can just 
#     import environments
#     current_env = environments.whatever
mooring_field = MooringField()