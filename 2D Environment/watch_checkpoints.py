# watch_checkpoints.py
# Watch a trained agent attempt to moor, with the same dials as main.py's
# manual test -- wheel, lever, tachometer -- driven by the AI's chosen
# action instead of your mouse.
#
# Run: python watch_checkpoints.py sac_checkpoint_500000.pt
# Keys: 1 / 2 / 4 = speed, E = toggle deterministic (evaluate) vs sampled actions

import sys
import math
import pygame
import torch

from rl_env import MooringEnv
from sac_neural_network import SACAgent, OBSERVATION_DIMENSION, ACTION_DIMENSION
from steering_wheel import SteeringWheel, TURNS_TO_LOCK
from throttle_lever import ThrottleLever
from tachometer import Tachometer

checkpoint_path = sys.argv[1]
device = "cpu" if torch.cuda.is_available() else "cpu"

env = MooringEnv()
agent = SACAgent(OBSERVATION_DIMENSION, ACTION_DIMENSION, device=device)

checkpoint = torch.load(checkpoint_path, map_location=device)
agent.actor.load_state_dict(checkpoint["actor"])
agent.critic1.load_state_dict(checkpoint["critic1"])
agent.critic2.load_state_dict(checkpoint["critic2"])
agent.log_alpha = checkpoint["log_alpha"]

pygame.init()
window_screen = pygame.display.set_mode((1000, 800))
clock = pygame.time.Clock()

wheel = SteeringWheel(900, 700, 100)
lever = ThrottleLever(700, 700, 100, 50)
tach = Tachometer(500, 700)

# NEW: speed, evaluate mode, and tally
speed_multiplier = 40
evaluate_mode = False
docks, episodes = 0, 0

def update_caption():  # NEW
    mode = "deterministic" if evaluate_mode else "sampled"
    pygame.display.set_caption(
        f"{checkpoint_path} | {speed_multiplier}x | {mode} | {docks}/{episodes} docked"
    )

update_caption()

observation, info = env.reset()
throttle_action, steering_action = 0.0, 0.0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # NEW: run several physics steps per drawn frame
    for _ in range(speed_multiplier):
        action = agent.select_action(observation, evaluate=evaluate_mode)
        throttle_action, steering_action = float(action[0]), float(action[1])
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            episodes += 1
            if info["outcome"] == "docked":
                docks += 1
            print(f"Episode {episodes}: {info['outcome']} | {docks}/{episodes} docked")
            update_caption()
            observation, info = env.reset()
            break  # start the next episode cleanly on the next frame

    # Drive the dials to show what the AI chose -- same math as the wheel's
    # and lever's own intensity methods, just run backwards (value -> angle
    # instead of angle -> value), since nothing is being dragged here.
    wheel.angle = steering_action * (TURNS_TO_LOCK * 2 * math.pi)
    lever.angle = throttle_action * lever.max_angle
    tach.update(throttle_action)

    window_screen.fill((40, 90, 130))
    env.env.draw_env(window_screen)
    env.boat.draw_boat(window_screen)
    wheel.draw_wheel(window_screen)
    lever.draw_lever(window_screen)
    tach.draw(window_screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()