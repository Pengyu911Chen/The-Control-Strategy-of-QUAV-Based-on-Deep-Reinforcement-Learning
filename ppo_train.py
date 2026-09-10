# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import gym

import numpy as np

import torch
from ppo_agent import PPOAgent
import time
import os.path



# environment is gym. After that ,it's replaced by mujoco
scenario="Pendulum-v1"
env= gym.make(scenario)

NUM_episode= 3000
NUM_step=200

state_dim= env.observation_space.shape[0] # Import dimensions of state from environment
action_dim= env.action_space.shape[0]

Batch_size = 25
UPDATE_INTERVAL = 50

# Direction for saving models
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + "/models/"
timestamp = time.strftime(("Y%m%d%H%M&S"))

# Agent : Input will be assigned as state acquired from env and Output will be action done by object
agent = PPOAgent(state_dim , action_dim , Batch_size) # TODO

REWARD_BUFFER = np.empty(shape = NUM_episode)

best_reward = -2000 # This parament should be noticeable!!!!!!!!!!!!!!!!!!!!!!

for episode_i in range(NUM_episode):
    state , other = env.reset() # reset funcation has two output return value: Observation and information
    done = False # To the beginning, we should define the ending of the code. But it should not end at first episode.
    episode_reward = 0

    for step_i in range(NUM_step):
       action , value = agent.get_action(state) # TODO
       next_state , reward  , truncated , info , done = env.step(action)
       episode_reward += reward
       
       done = True if (episode_i + 1) == NUM_step else False # Judge the ending point: Done means the nums of Episode have achieved
       agent.replay_buffer.add_memo(state , action , reward , value , done) # restore the paraments into buffer
       state = next_state

       if (step_i + 1) % UPDATE_INTERVAL == 0 or (step_i + 1)== NUM_step:
          agent.update() # TODO we need update the strategy in this loop
    
    if episode_reward >= -100 and episode_reward > best_reward: # the reward of Pendulum env always falls down to the Zero. Therefore -100 is a great reward. we need save it
       best_reward = episode_reward # avoid the initial update to the reward
       agent.save_policy()
       torch.save(agent.actor.state_dict() , model + f"ppo_actor_{timestamp}.pth")
       print(f"Best reward:{best_reward}")
    
    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode:{episode_i}, Reward:{round(episode_reward,2)}")

env.close()