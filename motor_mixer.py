# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

# https://zhuanlan.zhihu.com/p/91305836
import numpy as np

class Mixer:
    def __init__(self):
        self.Ct  = 3.25e-4
        self.Cd  = 7.9379e-6
        self.L   = 0.065/2.0
        self.max_thrust  = 0.1573
        self.max_torque  = 3.842e-03
        self.max_speed   = 22
        self.mat = np.array([
            [self.Ct, self.Ct, self.Ct, self.Ct],                                   # F total
            [self.Ct*self.L, -self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L],     # Mx + - - +
            [-self.Ct*self.L, -self.Ct*self.L, self.Ct*self.L, self.Ct*self.L],     # My - - + +
            [-self.Cd, self.Cd, -self.Cd, self.Cd]                                  # Mz - + - +
        ])
        self.inv_mat = np.linalg.inv(self.mat)

    def calculate(self, thrust, mx, my, mz):
        Mx, My = mx, my  # Copy
        Mz = 0
        control_input = np.array([thrust, Mx, My, Mz])
        motor_speed_squ = self.inv_mat @ control_input
        max_value = np.max(motor_speed_squ)
        min_value = np.min(motor_speed_squ)
        ref_value = np.sum(motor_speed_squ) / 4.0
        max_trim_scale = 1.0
        min_trim_scale = 1.0
        if max_value > self.max_speed **2:
            max_trim_scale = (self.max_speed ** 2 - ref_value)/(max_value - ref_value)
        if min_value < 0:
            min_trim_scale = (ref_value)/(ref_value - min_value)
        scale = min(max_trim_scale, min_trim_scale)
        Mx = Mx * scale  
        My = My * scale
        control_input = np.array([thrust, Mx, My, Mz])
        motor_speed_squ = self.inv_mat @ control_input
        if scale < 1.0:
            motor_speed_squ = np.abs(motor_speed_squ)
            return np.sqrt(motor_speed_squ)
        else:
            Mz = mz
            control_input_withz = np.array([thrust, Mx, My, Mz])
            motor_speed_squ_withz = self.inv_mat @ control_input_withz
            max_value = np.max(motor_speed_squ_withz)
            min_value = np.min(motor_speed_squ_withz)
            max_index = np.argmax(motor_speed_squ_withz)
            min_index = np.argmin(motor_speed_squ_withz)
            max_trim_scale_z = 1.0
            min_trim_scale_z = 1.0
            if max_value > self.max_speed **2:
                max_trim_scale_z = (self.max_speed ** 2 - motor_speed_squ[max_index])/(max_value - motor_speed_squ[max_index])
            if min_value < 0:
                min_trim_scale_z = (motor_speed_squ[min_index])/(motor_speed_squ[min_index] - min_value)
            scale_z = min(max_trim_scale_z, min_trim_scale_z)
            Mz = Mz * scale_z
            control_input_withz = np.array([thrust, Mx, My, Mz])
            motor_speed_squ_withz = self.inv_mat @ control_input_withz
            motor_speed_squ = np.abs(motor_speed_squ)
            motor = np.array([thrust, mx, my, mz])
            motor_speed_squ = np.abs(motor_speed_squ)
            if np.any(motor_speed_squ_withz < 0):
              print("motor_speed_squ_withz contains negative values.")
              print("control vector: ",control_input_withz)
              print("result: ", motor_speed_squ_withz)
              raise ValueError("Negative squared motor speed.")
            return np.sqrt(motor_speed_squ_withz)


if __name__ == '__main__':
    thrust = 0.2398347
    Mx =  0.00111288
    My =  0.0016691
    Mz = np.inf

    mixer = Mixer()
    motor_speed = mixer.calculate(thrust, Mx, My, Mz)
    print(f"Motor Speed:{motor_speed}")
