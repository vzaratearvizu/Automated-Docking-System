# Runs SAC against MooringEnv, headless (no pygame), with periodic
# checkpoints so we can watch progress separately via watch_trained.py.

import numpy
import torch

from rl_env import MooringEnv
from sac_neural_network import SACAgent, ReplayBuffer, OBSERVATION_DIMENSION, ACTION_DIMENSION

# Stores some amount of experiences/steps. Can be any number that may seem fit to work
BUFFER_CAPACITY = 500000
BATCH_SIZE = 800
# Pure random actions first, to seed the buffer so it has something to learn from
WARMUP_STEPS = 2400
TOTAL_STEPS = 2000000
CHECKPOINT_EVERY = 250000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

env = MooringEnv()
agent = SACAgent(OBSERVATION_DIMENSION, ACTION_DIMENSION, device=device)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY, OBSERVATION_DIMENSION, ACTION_DIMENSION)

# We don't need info, just good to see what it returns there
observation, info = env.reset()
episode_reward = 0.0
episode_count = 0

best_episode_reward = float("-inf")

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
    if step >= WARMUP_STEPS and len(replay_buffer) >= BATCH_SIZE and step % 8 == 0:
        agent.update(replay_buffer, BATCH_SIZE)

    if terminated or truncated:
        episode_count += 1
        print(f"Episode {episode_count} | step {step} | reward {episode_reward:.2f} | outcome {info['outcome']}")

        if episode_reward > 0:
            best_episode_reward = episode_reward
            torch.save({
                "actor": agent.actor.state_dict(),
                "critic1": agent.critic1.state_dict(),
                "critic2": agent.critic2.state_dict(),
                "log_alpha": agent.log_alpha,
                }, f"sac_checkpoint_best_{episode_reward:.2f}.pt")
            print(f"  New best! Saved sac_checkpoint_best_{episode_reward:.2f}.pt")

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
        }, path)
        print(f"Saved checkpoint: {path}")

print("Training complete.")