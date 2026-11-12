# watch_checkpoints.py
# Watch a trained agent attempt to moor, with the same dials as main.py's
# manual test -- wheel, lever, tachometer -- driven by the AI's chosen
# action instead of your mouse.
#
# Run: python watch_trained.py sac_checkpoint_50000.pt

import sys
import math
import pygame
import torch

from rl_env import MooringEnv
from sac_neural_network import SACAgent, OBSERVATION_DIMENSION, ACTION_DIMENSION  # match your actual filename
from steering_wheel import SteeringWheel, TURNS_TO_LOCK
from throttle_lever import ThrottleLever
from tachometer import Tachometer

checkpoint_path = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

env = MooringEnv()
agent = SACAgent(OBSERVATION_DIMENSION, ACTION_DIMENSION, device=device)

checkpoint = torch.load(checkpoint_path, map_location=device)
agent.actor.load_state_dict(checkpoint["actor"])
agent.critic1.load_state_dict(checkpoint["critic1"])
agent.critic2.load_state_dict(checkpoint["critic2"])
agent.log_alpha = checkpoint["log_alpha"]

pygame.init()
window_screen = pygame.display.set_mode((1000, 800))
pygame.display.set_caption("Watching Trained Agent")
clock = pygame.time.Clock()

wheel = SteeringWheel(900, 700, 100)
lever = ThrottleLever(700, 700, 100, 50)
tach = Tachometer(500, 700)

observation, info = env.reset()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = agent.select_action(observation, evaluate=True)
    throttle_action, steering_action = float(action[0]), float(action[1])
    observation, reward, terminated, truncated, info = env.step(action)

    # Drive the dials to show what the AI chose -- same math as the wheel's
    # and lever's own intensity methods, just run backwards (value -> angle
    # instead of angle -> value), since nothing is being dragged here.
    wheel.angle = steering_action * (TURNS_TO_LOCK * 2 * math.pi)
    lever.angle = throttle_action * lever.max_angle
    tach.update(throttle_action)

    if terminated or truncated:
        print(f"Episode ended: outcome={info['outcome']}")
        observation, info = env.reset()

    window_screen.fill((40, 90, 130))
    env.env.draw_env(window_screen)
    env.boat.draw_boat(window_screen)
    wheel.draw_wheel(window_screen)
    lever.draw_lever(window_screen)
    tach.draw(window_screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()