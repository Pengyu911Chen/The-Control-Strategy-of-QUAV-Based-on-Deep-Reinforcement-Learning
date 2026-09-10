# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

from torch import nn
import torch
from torch.distributions import Normal
import torch.optim as optim
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device}")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=6, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x))
        std = self.softplus(self.fc_std(x)) + 1e-3
        return mean, std

    def select_action(self, s):
        with torch.no_grad():
            mu, sigma = self.forward(s)
            normal_dist = Normal(mu, sigma)
            action = normal_dist.sample()
            action = action.clamp(-1.0, 1.0)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class ReplayMemory:
    def __init__(self, batch_size):
        self.state = []
        self.action = []
        self.reward = []
        self.value = []
        self.done = []
        self.BATCH_SIZE = batch_size

    def add_memo(self, state, action, reward, value, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.value.append(value)

    def sample(self):
        num_state = len(self.state)
        batch_start_points = np.arange(0, num_state, self.BATCH_SIZE)
        memory_indices = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(memory_indices)
        batches = [memory_indices[i:i + self.BATCH_SIZE] for i in batch_start_points]

        return np.array(self.state), \
               np.array(self.action), \
               np.array(self.reward), \
               np.array(self.value), \
               np.array(self.done), \
               batches

    def clear_memo(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.value = []


class PPOAgent:
    def __init__(self, state_dim, action_dim=6, batch_size=64):
        self.LearningRate_ACTOR = 2e-3
        self.LearningRate_CRITIC = 1e-3
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.NUM_EPOCH = 10
        self.EPSILON_CLIP = 0.2

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LearningRate_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LearningRate_CRITIC)
        self.replay_buffer = ReplayMemory(batch_size)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor.select_action(state)
        value = self.critic.forward(state)
        return action.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

        memo_states, memo_actions, memo_rewards, memo_values, memo_dones, batches = self.replay_buffer.sample()
        T = len(memo_rewards)
        memo_advantages = np.zeros(T, dtype=np.float32)

        for t in range(T):
            discount = 1
            a_t = 0
            for k in range(t, T - 1):
                a_t += discount * (memo_rewards[k] + self.GAMMA * memo_values[k + 1] * (1 - int(memo_dones[k])) - memo_values[k])
                discount *= self.GAMMA * self.LAMBDA
            memo_advantages[t] = a_t

        with torch.no_grad():
            memo_advantages_tensor = torch.tensor(memo_advantages).unsqueeze(1).to(device)
            memo_values_tensor = torch.tensor(memo_values).to(device)
        memo_states_tensor = torch.FloatTensor(memo_states).to(device)
        memo_actions_tensor = torch.FloatTensor(memo_actions).to(device)

        for batch in batches:
            with torch.no_grad():
                old_mu, old_sigma = self.old_actor(memo_states_tensor[batch])
                old_pi = Normal(old_mu, old_sigma)
            batch_old_probs_tensor = old_pi.log_prob(memo_actions_tensor[batch])

            mu, sigma = self.actor(memo_states_tensor[batch])
            pi = Normal(mu, sigma)
            batch_probs_tensor = pi.log_prob(memo_actions_tensor[batch])

            ratio = torch.exp(batch_probs_tensor - batch_old_probs_tensor)
            surr1 = ratio * memo_advantages_tensor[batch]
            surr2 = torch.clamp(ratio, 1 - self.EPSILON_CLIP, 1 + self.EPSILON_CLIP) * memo_advantages_tensor[batch]
            actor_loss = -torch.min(surr1, surr2).mean()

            batch_returns = memo_advantages_tensor[batch] + memo_values_tensor[batch]
            batch_old_values = self.critic(memo_states_tensor[batch])
            critic_loss = nn.MSELoss()(batch_old_values, batch_returns)

            print(f"Critic Loss: {critic_loss.item():.6f}")

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.replay_buffer.clear_memo()

    def save_policy(self):
        torch.save(self.actor.state_dict(), f"PPO_Policy_Drone_Control.pt")
