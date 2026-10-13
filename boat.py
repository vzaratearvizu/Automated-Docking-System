# This creates the boat that is currently just a rectangle. All appearance, no physics or collisions.
# Once the project is functionally complete, can be altered to be closer to an actual boat shape.

import math
import pygame

# Pixels per frame at full throttle
MAX_SPEED = 3.0
# How quickly actual speed chases target speed (0-1)
# Smaller = slower to accelerate, more lag
THROTTLE_RESPONSE = 0.0002

class Boat:
    def __init__(self, x_pos, y_pos, width, height):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.height = height
        self.heading = 0.0  # radians, 0 = pointing right. Not wired to the wheel yet.
        self.speed = 0.0    # current actual speed -- changes gradually, not instantly

    def update(self, throttle):
        """throttle: -1.0 (full reverse) to 1.0 (full forward)."""
        target_speed = throttle * MAX_SPEED

        # Speed doesn't jump straight to target_speed -- it chases it a little
        # bit each frame, same idea as a real engine spooling up/down rather
        # than snapping to max RPM instantly.
        self.speed += (target_speed - self.speed) * THROTTLE_RESPONSE

        self.x_pos += self.speed * -math.sin(self.heading)
        self.y_pos += self.speed * -math.cos(self.heading)

    def draw_boat(self, surface):
        rect = pygame.Rect(self.x_pos, self.y_pos, self.width, self.height)
        # The body of the boat
        pygame.draw.rect(surface, (227, 119, 111), rect)
        # Outline of the boat
        pygame.draw.rect(surface, (255, 255, 255), rect, width=2) 
