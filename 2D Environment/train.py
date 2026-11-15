# Runs SAC against MooringEnv, headless (no pygame), with periodic
# checkpoints so we can watch progress separately via watch_checkpoints.py.

import numpy
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from rl_env import MooringEnv
from sac_neural_network import SACAgent, ReplayBuffer, OBSERVATION_DIMENSION, ACTION_DIMENSION

# Stores some amount of experiences/steps. Can be any number that may seem fit to work
BUFFER_CAPACITY = 250000
BATCH_SIZE = 256
# Pure random actions first, to seed the buffer so it has something to learn from
WARMUP_STEPS = 20000
TOTAL_STEPS = 4000000
CHECKPOINT_EVERY = 250000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# creates a ./runs/<timestamp> folder automatically
writer = SummaryWriter()

env = MooringEnv()
agent = SACAgent(OBSERVATION_DIMENSION, ACTION_DIMENSION, device=device)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY, OBSERVATION_DIMENSION, ACTION_DIMENSION)

# We don't need info, just good to see what it returns there
observation, info = env.reset()
episode_reward = 0.0
episode_count = 0
dock_count = 0 
recent_docks = deque(maxlen=20)
SAVE_RATE_THRESHOLD = 0.70
best_saved_rate = 0

for step in range(1, TOTAL_STEPS + 1):
    if step < WARMUP_STEPS:
        # Be completely random
        action = numpy.random.uniform(-1, 1, size=ACTION_DIMENSION).astype(numpy.float32)
    else:
        # The agent will begin to make educated guesses on an appropriate action to make given the observation
        action = agent.select_action(observation)

    # Make the action given the observation
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Only a TRUE ending (docked) should zero out the bootstrap -- a timeout
    # is us cutting the episode short, not the task itself ending.
    done = terminated
    replay_buffer.add(observation, action, reward, next_observation, float(done))

    observation = next_observation
    episode_reward += reward

    # A smaller ratio (more steps until updating) means slower learning
    # Bigger ratio means early risk of early overfitting, also called primacy bias
    if step >= WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE and step % 1 == 0:
        metrics = agent.update(replay_buffer, BATCH_SIZE)
        if step % 1000 == 0:
            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, step)

    if terminated or truncated:
        episode_count += 1
        docked = info["outcome"] == "docked"
        recent_docks.append(1 if docked else 0)
        if docked:
            dock_count += 1
        dock_rate = sum(recent_docks) / len(recent_docks)

        if len(recent_docks) == recent_docks.maxlen and dock_rate >= SAVE_RATE_THRESHOLD and dock_rate > best_saved_rate:
            best_saved_rate = dock_rate
            torch.save({
                "actor": agent.actor.state_dict(),
                "critic1": agent.critic1.state_dict(),
                "critic2": agent.critic2.state_dict(),
                "log_alpha": agent.log_alpha,
                "step": step
            }, f"sac_checkpoint_rate_{dock_rate:.2f}_step{step}.pt")
            print(f"  New best dock rate {dock_rate:.2f} | Saved checkpoint.")

        writer.add_scalar("episode/reward", episode_reward, episode_count)
        writer.add_scalar("episode/dock_count", dock_count, episode_count)
        writer.add_scalar("episode/dock_rate_last_20", dock_rate, episode_count)
        print(f"Episode {episode_count} | step {step} | reward {episode_reward:.2f} | outcome {info['outcome']} | docks {dock_count} | last 20 {dock_rate:.2f}")

        # info not needed, just good to see what goes there
        observation, info = env.reset()
        episode_reward = 0.0

    if step % CHECKPOINT_EVERY == 0:
        path = f"sac_checkpoint_{step}.pt"
        torch.save({
            "actor": agent.actor.state_dict(),
            "critic1": agent.critic1.state_dict(),
            "critic2": agent.critic2.state_dict(),
            "log_alpha": agent.log_alpha,
            "step": step
        }, path)
        print(f"Saved checkpoint: {path}")

writer.close()
print("Training complete.")