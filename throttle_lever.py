# A side-view throttle lever since it's more informative seeing the angle of throttle rather than a top down view

import math
import pygame

class ThrottleLever:
    def __init__(self, pivot_x, pivot_y, length, max_angle_deg):
        self.pivot_x = pivot_x
        self.pivot_y = pivot_y
        self.length = length
        # The hard stop in each direction for the lever
        self.max_angle = math.radians(max_angle_deg)
        # In radians, 0 is straight up and neutral
        self.angle = 0.0
        self.dragging = False
        self.font = pygame.font.SysFont(None, 24)

    def throttle_intensity(self):
        # -1.0 will be full reverse and +1.0 will be full forward. Zero is neutral
        # Since the center-up is zero, one direction will be positive while the other is negative
        return self.angle / self.max_angle

    def _handle_pos(self):
        # math.sin(whatever) gives the x-coordinate it's in the circle
        # Since the the zero angle is north of the circle, sin(0) provides 0 as the coordinate
        handle_x = self.pivot_x + self.length * math.sin(self.angle)
        # math.cos(whatever) gives the y-coordinate it's in on the circle
        # Since the the zero angle is north of the circle, cos(0) provides 1 as the coordinate
        handle_y = self.pivot_y - self.length * math.cos(self.angle)
        return handle_x, handle_y

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x, handle_y = self._handle_pos()
            mouse_x, mouse_y = event.pos
            distance = math.hypot(mouse_x - handle_x, mouse_y - handle_y)
            # distance will be the radius clicked from the center of the handle
            if distance <= 10:
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x, mouse_y = event.pos
            distance_x = mouse_x - self.pivot_x
            distance_y = mouse_y - self.pivot_y
            # Same explanation as the wheel
            raw_angle = math.atan2(distance_x, -distance_y)
            # Stops the angle from reaching past a point because it's a lever
            self.angle = max(-self.max_angle, min(self.max_angle, raw_angle))

    def draw_lever(self, surface):
        # Pivot / hinge base
        pygame.draw.circle(surface, (70, 70, 75), (self.pivot_x, self.pivot_y), 10)

        # The slot the lever travels through, so you can see its full range
        rev_x = self.pivot_x + self.length * math.sin(-self.max_angle)
        rev_y = self.pivot_y - self.length * math.cos(-self.max_angle)
        fwd_x = self.pivot_x + self.length * math.sin(self.max_angle)
        fwd_y = self.pivot_y - self.length * math.cos(self.max_angle)
        pygame.draw.line(surface, (60, 60, 65), (rev_x, rev_y), (fwd_x, fwd_y), 3)

        # The arm itself, from pivot to handle
        tip_x, tip_y = self._handle_pos()
        pygame.draw.line(surface, (150, 150, 155), (self.pivot_x, self.pivot_y), (tip_x, tip_y), 5)
        pygame.draw.circle(surface, (80, 180, 220), (int(tip_x), int(tip_y)), 12)

        # Labels: REV / N / FWD at their actual positions along the arc
        surface.blit(self.font.render("FWD", True, (230, 230, 230)), (fwd_x - 10, fwd_y - 20))
        surface.blit(self.font.render("N", True, (230, 230, 230)), (self.pivot_x - 5, self.pivot_y - self.length - 20))
        surface.blit(self.font.render("REV", True, (230, 230, 230)), (rev_x - 15, rev_y - 20))

        value_text = self.font.render(f"throttle {self.throttle_intensity():+.2f}", True, (230, 230, 230))
        surface.blit(value_text, (self.pivot_x - 45, self.pivot_y + 20))