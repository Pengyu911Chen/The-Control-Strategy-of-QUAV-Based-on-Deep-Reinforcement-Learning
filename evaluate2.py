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
from scipy.spatial.transform import Rotation as R

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

    target_quat = np.array([1, 0, 1, 0])
    target_quat = target_quat / np.linalg.norm(target_quat)
    target_euler = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_euler('xyz', degrees=True)

    trajectory = []
    actual_euler_list = []
    target_euler_list = []

    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        mean, std = model(obs_tensor)
        normal_dist = Normal(mean, std)
        action = normal_dist.sample().clamp(-1.0, 1.0).numpy()[0]

        obs, reward, done, info = env.step(action)

        quat = obs[3:7]
        quat = quat / np.linalg.norm(quat)
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler('xyz', degrees=True)

        actual_euler_list.append(euler)
        target_euler_list.append(target_euler)
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

    actual_euler_array = np.array(actual_euler_list)
    target_euler_array = np.array(target_euler_list)

    plt.figure(figsize=(12, 6))
    titles = ['Roll (X axis)', 'Pitch (Y axis)', 'Yaw (Z axis)']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(actual_euler_array[:, i], label='Actual attitude')
        plt.plot(target_euler_array[:, i], '--', label='Target attitude')
        plt.ylabel(titles[i] + ' (deg)')
        plt.legend()
        plt.grid(True)

    plt.xlabel('Time step')
    plt.suptitle('Quadrotor attitude tracking in Euler angles')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
