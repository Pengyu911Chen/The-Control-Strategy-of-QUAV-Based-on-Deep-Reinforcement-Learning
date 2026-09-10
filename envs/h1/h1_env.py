# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import os
import copy
import numpy as np
import transforms3d as tf3
import collections
import mujoco
from pathlib import Path

from robots.robot_base import QuadrotorBase
from envs.common import mujoco_env
from envs.common import robot_interface
from envs.common import config_builder
from target import target

target_pos = [0.5, 0.5 , 0.5]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Task:
    def __init__(self, client):
        self._client = client
        #self.out_of_bounds = False
        self._success_threshold = 0.05
        self._boundary_limit = 2.0
        self._min_altitude = 3
        self._max_steps = 1000
        self._step_count = 0
        self.target=target()
        
    def calc_reward(self, prev_action, action):
        
        pos = self._client.get_qpos()[:3]
        pos_error = target_pos - pos
        pos_reward = np.exp(-5 * np.linalg.norm(pos_error) ** 2)
        quat = self._client.get_qpos()[3:7]
        euler = tf3.euler.quat2euler(quat)
        attitude_error = np.linalg.norm(euler)

        ang_vel = self._client.get_qvel()[3:6]
        ang_vel_error = np.linalg.norm(ang_vel)
        
        yaw = euler[2]
        yaw_error = yaw
        yaw_reward = np.exp(-2.0 * yaw_error**2)
        
        action_penalty = np.exp(-0.5 * np.linalg.norm(action - prev_action))
        
        roll, pitch, yaw = euler
        roll_limit = np.radians(45)
        pitch_limit = np.radians(45)

        over_limit_penalty = 0
        out_of_bounds = False
        if abs(roll) > roll_limit or abs(pitch) > pitch_limit:
            over_limit_penalty = -5.0
            #out_of_bounds = True
        
        if np.linalg.norm(pos_error) < 0.15 and np.linalg.norm(ang_vel) < 0.15:
          success_bonus = +50.0
        else:
          success_bonus = 0.0
        
        total_reward = (
        5.0 * pos_reward +
        1.0 * yaw_reward +
        0.5 * action_penalty +
        over_limit_penalty +
        success_bonus
        )

        #self.out_of_bounds = out_of_bounds

        return float(total_reward) 
    

    def step(self):
        pass

    def substep(self):
        pass

    def done1(self):
     pos = self._client.get_qpos()[:3]
     distance = np.linalg.norm(target_pos - pos)
     return bool(distance < 1)

    def done(self):
      pos = self._client.get_qpos()[:3]
      
      distance_to_target = np.linalg.norm(target_pos - pos)

      reached_target = distance_to_target < self._success_threshold

      out_of_bounds = np.any(np.abs(pos) > self._boundary_limit)

      #low_altitude = pos[2] < self._min_altitude

      timeout = self._step_count >= self._max_steps

      return bool(reached_target or out_of_bounds  or timeout)


    def reset(self):
        pass



class Quadrotor_Env(mujoco_env.MujocoEnv):
    def __init__(self, path_to_yaml=None):
         ## Load CONFIG from yaml ##
        if path_to_yaml is None:
            path_to_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/base.yaml')

        self.cfg = config_builder.load_yaml(path_to_yaml)

        sim_dt = self.cfg.sim_dt
        control_dt = self.cfg.control_dt
        frame_skip = int(control_dt / sim_dt)
        
        model_path = str(PROJECT_ROOT / "crazyfile" / "scene.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, sim_dt, control_dt)

        self.interface = robot_interface.QuadMotorModel(self.model, self.data, None)
        
        self.task = Task(self.interface)
        self.robot = QuadrotorBase(control_dt, self.interface, self.task)

        self.action_space = np.zeros(4)
        self.observation_space = np.zeros(13)
        
        base_position = [0, 0, 0.1]
        base_orientation = [1, 0, 0,0]
        base_euler = tf3.euler.quat2euler(base_orientation)
        base_linear_velocity = [0,0,0]
        base_angular_velocity= [0,0,0]
        self.nominal_pose = base_position + base_orientation #+ base_linear_velocity + base_angular_velocity
        self.prev_prediction = np.zeros(4)
        self.history_len = self.cfg.obs_history_len
        t = self.data.time
        self.target = target()


    def get_obs(self):

        pos = self.interface.get_qpos()[:3]
       
        pos_err = (target_pos - pos)
        
        #print(self.target.simple_trajectory_3d(t))
        

        vel = self.interface.get_qvel()[:3]

        quat = self.interface.get_qpos()[3:7]
        euler = tf3.euler.quat2euler(quat)

        ang_vel = self.interface.get_qvel()[3:6]
        
        obs = np.concatenate([pos_err, quat, vel, ang_vel])
        
        return obs

    def step(self, action):
        prev_action = self.prev_prediction

        reward,done = self.robot.step(prev_action ,action)
        obs = self.get_obs()
       
        self.prev_prediction = action
        return obs, reward, done, {}
    
    
    def reset_model(self):
        
     init_qpos = self.nominal_pose.copy()
     init_qvel = [0.0] * (len(init_qpos)-1)

     self.set_state(
         np.asarray(init_qpos),
         np.asarray(init_qvel)
     )
    
     mujoco.mj_forward(self.model, self.data)

     self.task.reset()
     self.prev_prediction = np.zeros_like(self.prev_prediction)
     self.observation_history = collections.deque(maxlen=self.history_len)

     obs = self.get_obs()
     return obs



    #### randomizations and other utility functions ###########
    def randomize_perturb(self):
        frc_mag = self.cfg.perturbation.force_magnitude
        tau_mag = self.cfg.perturbation.force_magnitude
        for body in self.cfg.perturbation.bodies:
            self.data.body(body).xfrc_applied[:3] = np.random.uniform(-frc_mag, frc_mag, 3)
            self.data.body(body).xfrc_applied[3:] = np.random.uniform(-tau_mag, tau_mag, 3)
            if np.random.randint(2)==0:
                self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)

    def randomize_dyn(self):
        # dynamics randomization
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.leg_names]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0, 2)    # actuated joint frictionloss
            self.model.dof_damping[jnt] = np.random.uniform(0.02, 2)      # actuated joint damping

        # randomize com
        bodies = ["pelvis"]
        for legjoint in self.leg_names:
            bodyid = self.model.joint(legjoint).bodyid
            bodyname = self.model.body(bodyid).name
            bodies.append(bodyname)

        for body in bodies:
            default_mass = self.default_model.body(body).mass[0]
            default_ipos = self.default_model.body(body).ipos
            self.model.body(body).mass[0] = default_mass*np.random.uniform(0.95, 1.05)
            self.model.body(body).ipos = default_ipos + np.random.uniform(-0.01, 0.01, 3)

    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.distance = 5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 1.0

   
