# This file sets up the window which will display everything. 
# The creation of the objects such as the boat, water, land, and walkway will be pulled from
# a different file, so they won't be created in this file.

# NO AI TRAINING will be made here, it's all for visual testing and then also seeing what the AI is like
# at certain points.

# python -m pip install pygame-ce
import pygame

import environments
from boat import Boat
from steering_wheel import SteeringWheel
from tachometer import Tachometer
from throttle_lever import ThrottleLever

# Initializes all nmodules before they can be used. Like turning on a car engine.
pygame.init()

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800

window_screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Boat Docking Environment")

# Creates a clock object which keeps track of time in milliseconds
clock = pygame.time.Clock()
running = True

current_env = environments.mooring_field
boat = Boat(500, 500,40, 100)
steering_wheel = SteeringWheel(900, 700, 100)
tach = Tachometer(500, 700)
lever = ThrottleLever(700, 700, 100, 50)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        steering_wheel.handle_event(event)
        lever.handle_event(event)

    tach.update(lever.throttle_intensity())
    boat.update(lever.throttle_intensity(), steering_wheel.steering_intensity())

    # Water-blue background
    window_screen.fill((40, 90, 130))
    current_env.draw_env(window_screen)
    boat.draw_boat(window_screen)
    steering_wheel.draw_wheel(window_screen)
    tach.draw(window_screen)
    lever.draw_lever(window_screen)

    # flip() updates the visible screen with completed drawing (there's a hidden one actively drawing it)
    pygame.display.flip()

    # tick is a method name, not a unit of measurement. 
    # tick(x) = x frames per second. Never runs more times than specified, good for keeping things consistent.
    clock.tick(60)

pygame.quit()