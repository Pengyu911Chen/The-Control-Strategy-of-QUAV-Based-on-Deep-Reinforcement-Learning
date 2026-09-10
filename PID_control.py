# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import numpy as np
import geometry

class State:
    def __init__(self, pos, vel, quat, omega):
        self.position = pos     # x y z
        self.velocity = vel     # x y z
        self.quaternion = quat  # x y z w 
        self.omega = omega      # x y z

    def update(self, pos, vel, quat, omega):
        self.position = pos
        self.velocity = vel
        self.quaternion = quat
        self.omega = omega

class Control_Command:
    def __init__(self, thrust, wx, wy, wz):
        self.thrust = thrust
        self.angular = np.array([wx, wy, wz])


class PID_Controller:
    def __init__(self):
        self.goal_state: State = None
        self.current_state: State = None

        self.kp_x = 1.0
        self.kd_x = 1.0
        self.ki_x = 0.1

        self.kp_R = 1.0
        self.kd_w = 1.0
        self.ki_w = 0.1

        self.int_e_x = np.zeros(3)
        self.int_e_w = np.zeros(3)

        self.gravity = np.array([0.0, 0.0, -1.0])

    def reset_integrals(self):
        self.int_e_x[:] = 0
        self.int_e_w[:] = 0

    def update_linear_error(self,dt):
        if self.goal_state is None or self.current_state is None:
            print("Error: goal or current state is None")
            return
        # Errors
        e_x = np.zeros(3)
        e_v = np.zeros(3)
        # Position Error
        e_x[0] = self.current_state.position[0] - self.goal_state.position[0]
        e_x[1] = self.current_state.position[1] - self.goal_state.position[1]
        e_x[2] = self.current_state.position[2] - self.goal_state.position[2]
        # ex_norm = np.linalg.norm(e_x)
        # if ex_norm > self.ex_norm_max:
        # Velocity Error
        e_v[0] = self.current_state.velocity[0] - self.goal_state.velocity[0]
        e_v[1] = self.current_state.velocity[1] - self.goal_state.velocity[1]
        e_v[2] = self.current_state.velocity[2] - self.goal_state.velocity[2]
        self.int_e_x += e_x * dt
        return e_x, e_v


    def update_angular_error(self, trans_control, forward, dt):
        q_curr = geometry.GeoQuaternion(*self.current_state.quaternion)
        R_curr = q_curr.getRotationMatrix()

        goal_z = trans_control - self.gravity
        goal_z /= np.linalg.norm(goal_z) + 1e-6

        up = goal_z
        right_des = np.cross(forward, up)
        right_des /= np.linalg.norm(right_des) + 1e-6
        proj_fwd_des = np.cross(up, right_des)

        R_goal = np.column_stack((right_des, proj_fwd_des, up))
        thrust = np.linalg.norm(trans_control - self.gravity)

        e_R = 0.5 * geometry.veemap(R_goal.T @ R_curr - R_curr.T @ R_goal)

        w_curr = self.current_state.omega
        w_des = self.goal_state.omega
        e_w = w_curr - R_curr.T @ R_goal @ w_des

        self.int_e_w += e_R * dt

        return e_R, e_w, thrust

    def control_update(self, current_state: State, goal_state: State, dt, forward):
     self.current_state = current_state
     self.goal_state = goal_state
 
     e_x, e_v = self.update_linear_error(dt)
     x = -self.kp_x * e_x[0] - self.kd_x * e_v[0] - self.ki_x * self.int_e_x[0]
     y = -self.kp_x * e_x[1] - self.kd_x * e_v[1] - self.ki_x * self.int_e_x[1]
     z = -self.kp_x * e_x[2] - self.kd_x * e_v[2] - self.ki_x * self.int_e_x[2]
     trans_control = np.array([x, y, z])

     e_R, e_w, thrust = self.update_angular_error(trans_control, forward, dt)
     wx = -self.kp_R * e_R[0] - self.kd_w * e_w[0] - self.ki_w * self.int_e_w[0] + self.goal_state.omega[0]
     wy = -self.kp_R * e_R[1] - self.kd_w * e_w[1] - self.ki_w * self.int_e_w[1] + self.goal_state.omega[1]
     wz = -self.kp_R * e_R[2] - self.kd_w * e_w[2] - self.ki_w * self.int_e_w[2] + self.goal_state.omega[2]

     return Control_Command(thrust, wx, wy, wz)
