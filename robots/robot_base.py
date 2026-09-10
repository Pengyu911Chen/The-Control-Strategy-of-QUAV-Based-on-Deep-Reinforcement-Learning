# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import numpy as np
from motor_mixer import Mixer
from rl.algos.ppo import PPO
import torch
from envs.common.robot_interface import QuadMotorModel
import transforms3d as tf3

class QuadrotorBase:
    def __init__(self,dt, client, task, pdrand_k = 0 , motor_dyn_model=None):
        self.client = client
        self.task = task
        self.control_dt = dt

        

        self.pdrand_k = pdrand_k

        self.motor_dyn_model = motor_dyn_model
        self.frame_skip = int(self.control_dt / self.client.sim_dt())
        self.prev_action = np.zeros(4)
        self.prev_torque = np.zeros(3)
        


    

    def step(self,prev_action,action):
        
       
        for _ in range(self.frame_skip):
            motor_thrusts,mx,my,mz = self.client.map_action_to_physical(action)
            #print("Action:",motor_thrusts,mx,my,mz )
            self.client.set_motor_thrusts(motor_thrusts,mx,my,mz)
            self.client.step()
        
        self.task.step()

        reward = self.task.calc_reward(prev_action, motor_thrusts)
        done = self.task.done()

        self.prev_action = motor_thrusts
        self.prev_torque = np.array([mx,my,mz])

        return reward, done
    
   
    def step1(self,action):

        for _ in range(self.frame_skip):
            self.client.set_motor_thrusts_2(action)
           
            self.client.step()
        
        self.task.step()

        reward = self.task.calc_reward(self.prev_action, action)
        done = self.task.done()

        self.prev_action = action
        #self.prev_torque = torque_cmd

        return reward, done
    
    def mirror_observation(obs: torch.Tensor) -> torch.Tensor:
      """
      Mirror observation mapping for [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz].
      The y, vy, roll, wy, and yaw terms change sign.
      """
      mirrored_obs = [
      -0,   # ex
       1,   # ey
       2,   # ez
       3,   # qw
      -4,   # qx
       5,   # qy
      -6,   # qz
      -7,   # vx
       8,   # vy
       9,   # vz
     -10,   # wx
      11,   # wy
     -12    # wz
      ]

      return mirrored_obs


    def mirror_action(act: torch.Tensor) -> torch.Tensor:
      """
      Mirror action mapping for [ax, ay, az, wx, wy, wz].
      The ay, wy, and yaw-control terms change sign.
      """
      mirrored_act = [
       0,   # thrust: no change
      -1,   # wx inverted
       2,   # wy same
      -3    # wz inverted
       ]

      return mirrored_act
