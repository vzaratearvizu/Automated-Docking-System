# A gauge that shows the engine's RPM

import math
import pygame

# About the maximum RPM on one engine boat
MAX_RPM = 6000
# how quickly the needle catches up to the target RPM
RPM_RESPONSE = 0.017

class Tachometer:
    def __init__(self, center_x, center_y, radius=70):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

        # current displayed RPM, catches up gradually
        self.rpm = 0.0

        self.font = pygame.font.SysFont(None, 22)

        # Gauge sweeps from -120 degrees to +120 degrees
        self.min_angle = math.radians(-120)
        self.max_angle = math.radians(120)

    def update(self, throttle: float):
        # -1.0 to 1.0. Is determined by how much the throttle is moved by magnitude. So how hard the rpm is both forward and backward
        target_rpm = abs(throttle) * MAX_RPM
        self.rpm += (target_rpm - self.rpm) * RPM_RESPONSE

    def _needle_angle(self):
        fraction = self.rpm / MAX_RPM
        # minimum angle + where it is between the maximum and minimum angle to show where it is
        return self.min_angle + fraction * (self.max_angle - self.min_angle)

    def draw(self, surface):
        pygame.draw.circle(surface, (70, 70, 75), (self.center_x, self.center_y), self.radius, width=3)

        # Tick marks at 0, 1000, 2000... RPM around the arc
        for rpm_mark in range(0, MAX_RPM + 1, 1000):
            fraction = rpm_mark / MAX_RPM
            angle = self.min_angle + fraction * (self.max_angle - self.min_angle)
            inner_x = self.center_x + (self.radius - 10) * math.sin(angle)
            inner_y = self.center_y - (self.radius - 10) * math.cos(angle)
            outer_x = self.center_x + self.radius * math.sin(angle)
            outer_y = self.center_y - self.radius * math.cos(angle)
            pygame.draw.line(surface, (180, 180, 180), (inner_x, inner_y), (outer_x, outer_y), 2)

        # The needle itself
        angle = self._needle_angle()
        tip_x = self.center_x + (self.radius - 15) * math.sin(angle)
        tip_y = self.center_y - (self.radius - 15) * math.cos(angle)
        pygame.draw.line(surface, (220, 60, 60), (self.center_x, self.center_y), (tip_x, tip_y), 3)
        pygame.draw.circle(surface, (220, 60, 60), (self.center_x, self.center_y), 6)

        label = self.font.render(f"{int(self.rpm)} RPM", True, (230, 230, 230))
        label_rect = label.get_rect(center=(self.center_x, self.center_y + self.radius + 20))
        surface.blit(label, label_rect)