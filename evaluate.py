# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import torch
from Q_ppo_agent2 import Actor
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
from mpl_toolkits.mplot3d import Axes3D
import time

from quadrotor_env import QuadrotorEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_actor.pth", help="Path to a trained actor checkpoint.")
    args = parser.parse_args()

    env = QuadrotorEnv(render_mode="human")
    obs = env.reset()

    model = Actor(state_dim=obs.shape[0], action_dim=6)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    trajectory = []
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        mean, std = model(obs_tensor)
        normal_dist = Normal(mean, std)
        action = normal_dist.sample().clamp(-1.0, 1.0).numpy()[0]

        obs, reward, done, info = env.step(action)
        trajectory.append(obs[:3])
        time.sleep(0.01)
    env.close()

    trajectory = np.array(trajectory)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Flight trajectory', color='b')

    ax.set_title("Quadrotor 3D flight trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
