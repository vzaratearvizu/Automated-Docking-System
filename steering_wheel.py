# A wheel that can be clicked and dragged to rotate.

import math
import pygame

TURNS_TO_LOCK = 3

class SteeringWheel:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        # radians, 0 = straight up
        self.angle = 0.0 
        self.dragging = False
        self.last_mouse_angle = 0.0
        self.font = pygame.font.SysFont(None, 28)

    def _angle_from_center(self, mouse_x, mouse_y):
        dist_x = mouse_x - self.x
        dist_y = mouse_y - self.y
        # atan2 gives the angle of the mouse from the center of the wheel
        # pygame has y increase as you go down rather than decreasing, so we have to flip it.
        return math.atan2(dist_x, -dist_y)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # event.pos give mouse position as (x, y)
            mouse_x, mouse_y = event.pos
            # hypot (hypotenuse) seeing the distance the mouse was clicked to the radius of the circle.
            distance = math.hypot(mouse_x - self.x, mouse_y - self.y)

            if distance <= self.radius + 15:
                self.dragging = True
                self.last_mouse_angle = self._angle_from_center(mouse_x, mouse_y)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # event.pos give mouse position as (x, y)
            mouse_x, mouse_y = event.pos
            new_mouse_angle = self._angle_from_center(mouse_x, mouse_y)
            delta = new_mouse_angle - self.last_mouse_angle
            # (angle difference + pi) mod (2 * pi) - pi
            # Add to the current 180 degrees, mod by the full 360 degrees and then subtract pi to get the remaining amount
            # This is so we find the difference to add or subtract to the angle at this frame. 
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            # Helps with tracking the angle so it continues after 360 degrees rather than resetting to 0
            self.angle += delta
            self.last_mouse_angle = new_mouse_angle

    def steering_intensity(self):
        # -1.0 (full lock left) to 1.0 (full lock right), based on how many
        # of the TURNS_TO_LOCK rotations the wheel has actually been turned.
        max_wheel_angle = TURNS_TO_LOCK * 2 * math.pi  # 3 full turns, in radians
        return max(-1.0, min(1.0, self.angle / max_wheel_angle))

    def draw_wheel(self, surface):
        # Draws the wheel
        pygame.draw.circle(surface, (90, 90, 95), (self.x, self.y), self.radius, width=5)

        # A spoke/marker so you can SEE the angle changing as you drag
        tip_x = self.x + self.radius * math.sin(self.angle)
        tip_y = self.y - self.radius * math.cos(self.angle)
        pygame.draw.line(surface, (200, 200, 40), (self.x, self.y), (tip_x, tip_y), 2)
        pygame.draw.circle(surface, (200, 200, 40), (self.x, self.y), 2)

        degrees = math.degrees(self.angle)
        text_surface = self.font.render(f"{degrees:.0f}°", True, (230, 230, 230))
        text_rect = text_surface.get_rect(center=(self.x, self.y - self.radius - 20))
        surface.blit(text_surface, text_rect)