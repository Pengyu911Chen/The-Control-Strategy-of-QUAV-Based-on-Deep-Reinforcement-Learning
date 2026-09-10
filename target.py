# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import mujoco
import mujoco.viewer as viewer
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class target:
    def _init_(self):
        self.target_pos=[0,0,0.1]
    
    def simple_trajectory(self,t):
        wait_time = 1.5
        height = 0.7
        radius = 0.2
        speed = 0.1

        angle = 2 * np.pi * speed * (t - wait_time)
        if t < wait_time:
            return np.array([radius, 0, height]), np.array([0.0, 1.0, 0.0])
    
        pos = np.array([radius * np.cos(angle), radius * np.sin(angle), height])
        heading = np.array([-np.sin(angle), np.cos(angle), 0])
        return pos, heading

    def simple_trajectory_3d(self,t):
        wait_time = 2.5
        
        base_height = 0.5
        radius = 0.5
        
        angular_speed = 0.01
       

        angle = 2 * np.pi * angular_speed * (t - wait_time)
        angle = 2 * np.pi * angular_speed * (t - wait_time)

        if t < wait_time:
            return np.array([radius, 0, base_height])
    
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        #x = radius * angle
        #y = radius * angle
        #z = base_height + vertical_speed * (t - wait_time)
        z = base_height #+ 0.1 * np.sin(angle)

        pos = np.array([x, y, z])

        

        return pos

    def simple_trajectory_3d_by_index(self, s):
      """
      Generate a trajectory point from normalized path parameter s in [0, 1].
      """
      base_height = 0.5
      radius = 0.5
    
      angle = 2 * np.pi * s

      x = radius * np.cos(angle)
      y = radius * np.sin(angle)
      z = base_height

      pos = np.array([x, y, z])
      return pos
