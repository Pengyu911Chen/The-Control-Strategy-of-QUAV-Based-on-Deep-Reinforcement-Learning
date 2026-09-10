# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import numpy as np
import torch
from Q_ppo_agent2 import PPOAgent
import time
import os

from quadrotor_env import QuadrotorEnv

env = QuadrotorEnv(render=False)

NUM_EPISODES = 1000
NUM_STEPS = 200
BATCH_SIZE = 25
UPDATE_INTERVAL = 50

state_dim = env.observation_space.shape[0] # TODO
action_dim =  6 #env.action_space.shape[0] # TODO

agent = PPOAgent(state_dim, action_dim, BATCH_SIZE)

REWARD_BUFFER = np.empty(NUM_EPISODES)
best_reward = -1e6 # TODO

model_path = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(model_path, exist_ok=True)
timestamp = time.strftime("%Y%m%d%H%M%S")

for episode in range(NUM_EPISODES):
    state = env.reset() # TODO
    episode_reward = 0

    for step in range(NUM_STEPS):
        action, value = agent.get_action(state)
        next_state, reward, done, _ = env.step(action) # TODO

        agent.replay_buffer.add_memo(state, action, reward, value, done)
        state = next_state
        episode_reward += reward

        if (step + 1) % UPDATE_INTERVAL == 0 or step == NUM_STEPS - 1:
            agent.update()

        if done:
            break

    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy()
        torch.save(agent.actor.state_dict(), os.path.join(model_path, f"ppo_actor_{timestamp}.pth"))
        print(f"New best reward: {best_reward:.2f}")

    REWARD_BUFFER[episode] = episode_reward
    print(f"Episode {episode + 1} | Reward: {episode_reward:.2f}")

env.close()
