# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import numpy as np
import gym
from gym import spaces
import mujoco
from mujoco import viewer
import os
from motor_mixer import Mixer
import rewards
from scipy.spatial.transform import Rotation as R

GRAVITY = 9.8066        # m/s^2
MASS = 0.033            # kg
CT = 3.25e-4
CD = 7.9379e-6
MAX_THRUST = 0.1573
MAX_TORQUE = 3.842e-3
ARM_LENGTH = 0.065 / 2.0
TORQUE_SCALE = 0.001
DT = 0.001


def calc_motor_force(krpm): return CT * krpm**2

def calc_motor_speed_by_force(force):
    force = np.clip(force, 0, MAX_THRUST)
    return np.sqrt(force / CT)

def calc_motor_speed_by_torque(torque):
    torque = np.clip(torque, 0, MAX_TORQUE)
    return np.sqrt(torque / CD)

def calc_motor_input(krpm):
    krpm = np.clip(krpm, 0, 22)
    force = calc_motor_force(krpm)
    return np.clip(force / MAX_THRUST, 0, 1)


class QuadrotorEnv(gym.Env):
    def __init__(self, render=False, xml_path=None, render_mode=None):
        super(QuadrotorEnv, self).__init__()

        if render == "human" and render_mode is None:
            render_mode = "human"

        self.render_mode = render_mode

        base_dir = os.path.dirname(__file__)
        if xml_path is None:
            xml_path = os.path.join(base_dir, "crazyfile", "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        self.render_enabled = bool(render) or render_mode == "human"
        if self.render_enabled:
            self.viewer = viewer.launch_passive(self.model, self.data)

        self.action_dim = 6
        self.max_thrust = 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        obs_high = np.array([np.inf] * 13, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.target_pos = np.array([4.0, 3.0, 3.0])

        self.target_quat = np.array([1.0, 0.0, 1.0, 0.0])  # w, x, y, z

        self.max_steps = 200
        self.step_count = 0
        self.gravity = np.array([0.0, 0.0, -1.0])
        self.mixer = Mixer()

    def calc_motor_input(krpm):
     krpm = np.clip(krpm, 0, 22)
     force = calc_motor_force(krpm)
     return np.clip(force / MAX_THRUST, 0, 1)


    def step(self, action):
        #thrusts = (action + 1.0) / 2.0 * self.max_thrust
        #print("action:", action, "shape:", action.shape)

        trans_control = action[:3]
        thrusts = np.linalg.norm(trans_control - self.gravity)
        #thrusts = action[] * GRAVITY * MASS

        torque = action[3:] * TORQUE_SCALE

        self._apply_motor_forces(thrusts , torque[0],torque[1],torque[2])

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward1(obs)
        done = self._is_done(obs)
        self.step_count += 1

        if self.render_enabled:
            self.viewer.sync()

        return obs, reward, done, {}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        self.data.qpos[:3] = np.array([0.0, 0.0, 1.0]) + np.random.normal(0, 0.05, size=3)
        self.data.qvel[:] = np.random.normal(0, 0.01, size=self.data.qvel.shape)

        return self._get_obs()

    def render(self, mode="human"):
        if self.render_enabled and self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.render_enabled and self.viewer is not None:
            self.viewer.close()

    def _apply_motor_forces(self, thrusts, mx, my, mz):
        """Map motor thrusts to MuJoCo actuator controls."""

        motor_speeds = self.mixer.calculate(thrusts, mx,my,mz)
        for i, speed in enumerate(motor_speeds, 1):
          self.data.actuator(f'motor{i}').ctrl[0] = calc_motor_input(speed)

        #for i in range(self.motor_speeds):
        #    self.data.ctrl[i] = thrusts[i]

    def _get_obs(self):
        pos = self.data.qpos[:3] 
        quat = self.data.qpos[3:7]
        lin_vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        return np.concatenate([pos, lin_vel, quat, ang_vel])

    def _compute_reward(self, obs):
        """Reward based on distance to the target position."""
        pos = obs[0:3]
        dist = np.linalg.norm(pos - self.target_pos)

        reward = -dist
        return reward
    def _compute_reward3(self, obs):
    

     pos = obs[0:3]
     lin_vel = obs[3:6]
     quat = obs[6:10]
     ang_vel = obs[10:13]

     dist_to_target = np.linalg.norm(pos - self.target_pos)
     position_reward = - dist_to_target

    
     velocity_penalty = - 0.1 * np.linalg.norm(lin_vel)
     ang_velocity_penalty = - 0.1 * np.linalg.norm(ang_vel)

     epsilon = 1e-3
     decay_velocity_penalty = - 1.0 * np.linalg.norm(lin_vel) / (dist_to_target + epsilon)

     control_penalty = - 0.01 * np.linalg.norm(self.last_action) if hasattr(self, 'last_action') else 0.0

     done_bonus = 10.0 if dist_to_target < 0.01 else 0.0

     reward = (1.0 * position_reward +
              velocity_penalty +
              ang_velocity_penalty +
              decay_velocity_penalty +
              control_penalty +
              done_bonus)

     return reward
    def _compute_reward1(self, obs):
    

     quat = obs[6:10]
     ang_vel = obs[10:13]

     
     current_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
     target_rot = R.from_quat([self.target_quat[1], self.target_quat[2], self.target_quat[3], self.target_quat[0]])
     relative_rot = current_rot.inv() * target_rot
     angle_error = relative_rot.magnitude()

     ang_vel_error = np.linalg.norm(ang_vel)

     control_effort = np.linalg.norm(self.last_action) if hasattr(self, 'last_action') else 0.0

     reward = - (5.0 * angle_error + 0.5 * ang_vel_error + 0.05 * control_effort)

     return reward

    def _is_done(self, obs):
        pos = obs[0:3]
        if np.linalg.norm(pos - self.target_pos) < 0.05:
            return True
        if self.step_count >= self.max_steps:
            return True
        if pos[2] < 0.1 or pos[2] > 5.0:
            return True
        return False
