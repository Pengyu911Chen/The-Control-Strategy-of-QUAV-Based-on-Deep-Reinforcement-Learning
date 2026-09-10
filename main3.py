# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

from pathlib import Path

import mujoco
import mujoco.viewer as viewer
import numpy as np
from PID_control import PID_Controller, State
from motor_mixer import Mixer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PROJECT_ROOT = Path(__file__).resolve().parent
real_traj = []
target_traj = []

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

def simple_trajectory(t):
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

def simple_trajectory_3d(t):
    wait_time = 1.5
    base_height = 0.5
    radius = 0.5
    angular_speed = 0.5
    vertical_speed = 1

    angle = 2 * np.pi * angular_speed * (t - wait_time)
    
    if t < wait_time:
        return np.array([radius, 0, base_height]), np.array([0.0, 1.0, 0.0])
    
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    #z = base_height + vertical_speed * (t - wait_time)
    z = base_height #+ 0.1 * np.sin(angle)

    pos = np.array([x, y, z])

    dx = -radius * np.sin(angle) * 2 * np.pi * angular_speed
    dy = radius * np.cos(angle) * 2 * np.pi * angular_speed
    dz = vertical_speed
    heading = np.array([dx, dy, dz])
    heading /= np.linalg.norm(heading) + 1e-6

    return pos, heading



ctrl = PID_Controller()

ctrl.kp_x = 1
ctrl.kd_x = 0.3
ctrl.ki_x = 0.01

ctrl.kp_R = 10.0
ctrl.kd_w = 0.3
ctrl.ki_w = 0.05

mixer = Mixer()
log_count = 0

def control_callback(m, d):
    global log_count

    sd = d.sensordata
    quat = np.array([sd[7], sd[8], sd[9], sd[6]])   # x y z w
    omega = np.array([sd[0], sd[1], sd[2]])

    curr_pos = d.qpos
    curr_vel = d.qvel
    curr_state = State(curr_pos, curr_vel, quat, omega)

    goal_pos, goal_heading = simple_trajectory_3d(d.time)
    goal_state = State(goal_pos, np.zeros(3), np.array([0, 0, 0, 1]), np.zeros(3))

    command = ctrl.control_update(curr_state, goal_state, DT, goal_heading)
    
    thrust_N = command.thrust * GRAVITY * MASS
    torque_Nm= command.angular * TORQUE_SCALE

    motor_speeds = mixer.calculate(thrust_N, *torque_Nm)
    for i, speed in enumerate(motor_speeds, 1):
        d.actuator(f'motor{i}').ctrl[0] = calc_motor_input(speed)

    log_count += 1
    real_traj.append(curr_pos[:3].copy())
    target_traj.append(goal_pos.copy())

    if log_count >= 500:
        log_count = 0
        #print(f"Time: {d.time:.2f} s | Pos: {curr_pos[:3]} | Goal: {goal_pos} | Thrust: {thrust_N:.4f} N")
        print(f"Torque:{torque_Nm[:3]}")

def load_callback():
    mujoco.set_mjcb_control(None)
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "crazyfile" / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.set_mjcb_control(control_callback)
    return model, data

if __name__ == '__main__':
    try:
        viewer.launch(loader=load_callback)
    finally:
        real_np = np.array(real_traj)
        target_np = np.array(target_traj)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(real_np[:, 0], real_np[:, 1], real_np[:, 2], label='Actual Trajectory', color='b')
        ax.plot(target_np[:, 0], target_np[:, 1], target_np[:, 2], label='Target Trajectory', color='r', linestyle='--')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Tracking Performance')
        ax.legend()
        plt.show()
