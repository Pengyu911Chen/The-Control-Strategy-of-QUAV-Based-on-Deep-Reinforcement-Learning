# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import torch
import time
from pathlib import Path

import mujoco
import mujoco.viewer

import imageio
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class EvaluateEnv:
    def __init__(self, env, policy, args):
        self.env = env
        self.policy = policy
        self.ep_len = args.ep_len

        if args.out_dir is None:
            args.out_dir = Path(args.path.parent, "videos")

        video_outdir = Path(args.out_dir)
        try:
            Path.mkdir(video_outdir, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            video_fn = Path(video_outdir, args.path.stem + "-" + now + ".mp4")
            self.writer = imageio.get_writer(video_fn, fps=60)
        except Exception as e:
            print("Could not create video writer:", e)
            exit(-1)

    @torch.no_grad()
    def run(self):

        height = 480
        width = 640
        renderer = mujoco.Renderer(self.env.model, height, width)
        viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        frames = []
                # Initialize lists for recording position over time
        x_pos = []
        y_pos = []
        z_pos = []
        time_stamps = []

        # Make a camera.
        cam = viewer.cam
        mujoco.mjv_defaultCamera(cam)
        cam.elevation = -20
        cam.distance = 4

        reset_counter = 0
        observation = self.env.reset()
        while self.env.data.time < self.ep_len:
            #time.sleep(0.01)
            step_start = time.time()

            # forward pass and step
            raw = self.policy.forward(torch.tensor(observation, dtype=torch.float32), deterministic=True).detach().numpy()
            observation, reward, done, _ = self.env.step(raw.copy())

            # render scene
            cam.lookat = self.env.data.body(1).xpos.copy()
            renderer.update_scene(self.env.data, cam)
            pixels = renderer.render()
            frames.append(pixels)

            viewer.sync()

            if done and reset_counter < 3:
                observation = self.env.reset()
                reset_counter += 1

            time_until_next_step = max(
                0, self.env.frame_skip*self.env.model.opt.timestep - (time.time() - step_start))
            time.sleep(time_until_next_step)
            
            # Record position and time
            drone_pos = self.env.data.body(1).xpos.copy()  # Assuming body ID 1 is the drone
            x_pos.append(drone_pos[0])
            y_pos.append(drone_pos[1])
            z_pos.append(drone_pos[2])
            time_stamps.append(self.env.data.time)
        # Plot tracking results
        
        for frame in frames:
            self.writer.append_data(frame)
        self.writer.close()
        self.env.close()
        viewer.close()
       # Plot tracking results
       # Plot tracking results
        

       # Plot tracking results
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(time_stamps, x_pos, label='X Position')
        plt.axhline(0.5, color='r', linestyle='--', label='Target X')
        plt.ylabel("X")
        plt.legend()
        #plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(time_stamps, y_pos, label='Y Position')
        plt.axhline(0.5, color='r', linestyle='--', label='Target Y')
        plt.ylabel("Y")
        plt.legend()
        #plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(time_stamps, z_pos, label='Z Position')
        plt.axhline(0.6, color='r', linestyle='--', label='Target Z')
        plt.xlabel("Time [s]")
        plt.ylabel("Z")
        plt.legend()
        #plt.grid()

        plt.tight_layout()
        plt.savefig("tracking_plot.png")  # Save figure to current directory
        plt.close()

      