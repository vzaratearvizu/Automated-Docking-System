# This creates the boat that is currently just a rectangle. All appearance, no physics or collisions.
# Once the project is functionally complete, can be altered to be closer to an actual boat shape.

import math
import pygame

# Pixels per frame at full throttle
MAX_SPEED = 3.0
# How quickly actual speed chases target speed (0-1)
# Smaller = slower to accelerate, more lag
THROTTLE_RESPONSE = 0.0002
# radians per frame at full wheel lock
MAX_TURN_RATE = 0.03
# How quickly actual angular velocity chases the target turn rate (0-1)
TURN_RESPONSE = 0.033

class Boat:
    def __init__(self, x_pos, y_pos, width, height):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.height = height
        self.heading = 0.0  # radians, 0 = pointing right. Not wired to the wheel yet.
        self.speed = 0.0    # current actual speed -- changes gradually, not instantly
        self.angular_velocity = 0.0  # current actual turning rate -- also changes gradually

    def update(self, throttle, steering):
        """throttle: -1.0 (full reverse) to 1.0 (full forward)."""
        target_speed = throttle * MAX_SPEED

        # Speed doesn't jump straight to target_speed -- it chases it a little
        # bit each frame, same idea as a real engine spooling up/down rather
        # than snapping to max RPM instantly.
        self.speed += (target_speed - self.speed) * THROTTLE_RESPONSE

        target_turn_rate = steering * MAX_TURN_RATE * abs(throttle)
        self.angular_velocity += (target_turn_rate - self.angular_velocity) * TURN_RESPONSE
        self.heading += self.angular_velocity

        self.x_pos += self.speed * math.sin(self.heading)
        self.y_pos += self.speed * -math.cos(self.heading)

    def draw_boat(self, surface):
        # Draw the boat on its own small surface first, so it can be rotated
        # independently before being placed into the world.
        boat_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        rect = boat_surface.get_rect()
        pygame.draw.rect(boat_surface, (227, 119, 111), rect)
        pygame.draw.rect(boat_surface, (255, 255, 255), rect, width=2)

        # A small triangle marking the bow (front) -- without this, a plain
        # rectangle looks identical rotated 180 degrees, so you still
        # couldn't tell which end is actually facing forward.
        bow = [(self.width / 2, 0), (self.width / 2 - 8, 16), (self.width / 2 + 8, 16)]
        pygame.draw.polygon(boat_surface, (255, 220, 60), bow)

        # pygame.transform.rotate turns COUNTERclockwise for positive degrees,
        # but our heading convention is the opposite (0=up, positive=clockwise,
        # matching the wheel), so we negate when converting.
        degrees = -math.degrees(self.heading)
        rotated = pygame.transform.rotate(boat_surface, degrees)
        rotated_rect = rotated.get_rect(center=(self.x_pos, self.y_pos))
        surface.blit(rotated, rotated_rect)

    def reset(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.heading = 0.0
        self.speed = 0.0
        self.angular_velocity = 0.0