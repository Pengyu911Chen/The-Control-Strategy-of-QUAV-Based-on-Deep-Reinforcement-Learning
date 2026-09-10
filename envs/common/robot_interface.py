# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import os
import numpy as np
import transforms3d as tf3
import mujoco
import torch
import collections
from motor_mixer import Mixer

GRAVITY = 9.8066        # m/s^2
MASS = 0.033            # kg
CT = 3.25e-4
CD = 7.9379e-6
MAX_THRUST = 0.1573
MAX_TORQUE = 3.842e-3
ARM_LENGTH = 0.065 / 2.0
TORQUE_SCALE = 0.008
THRUST_SCALE = 0.2
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


class QuadMotorModel:
    def __init__(self, model, data, path_to_motor_nets=None):
        self.model = model
        self.data = data

        self.motor_names = ['motor1', 'motor2', 'motor3', 'motor4']
        self.step_counter = 0
        
        if path_to_motor_nets:
            self.load_motor_nets(path_to_motor_nets)
        self.mixer = Mixer()


    def calc_motor_speed_by_torque(self,torque):
        torque = np.clip(torque, 0, MAX_TORQUE)
        return np.sqrt(torque / CD)

    def load_motor_nets(self, path_to_motor_nets):
        self.motor_dyn_nets = {}
        for motor_name in self.motor_names:
            net_path = os.path.join(path_to_motor_nets, motor_name, "trained_jit.pth")
            if not os.path.exists(net_path):
                raise FileNotFoundError(f"Missing motor net for {motor_name}")
            net = torch.jit.load(net_path)
            net.eval()
            self.motor_dyn_nets[motor_name] = net

        self.cthrust_buffer = collections.deque(maxlen=25)  # control thrust
        self.omega_buffer = collections.deque(maxlen=25)    # motor speed or rotor angular velocity

    def get_motor_speeds(self,torque_cmd):
        motor_speed =  np.zeros[4]
        motor_speed = calc_motor_speed_by_torque(torque_cmd * TORQUE_SCALE)
        return motor_speed

    def motor_nets_forward(self, cmdThrusts,torque_cmd):
        if len(self.cthrust_buffer) < self.cthrust_buffer.maxlen:
            omega = self.get_motor_speeds(torque_cmd)
            self.omega_buffer.append(omega)
            self.cthrust_buffer.append(cmdThrusts)
            return cmdThrusts

        if (self.step_counter % 2) == 0:
            omega = self.get_motor_speeds(torque_cmd)
            self.omega_buffer.append(omega)
            self.cthrust_buffer.append(cmdThrusts)

        actual_thrusts = np.copy(cmdThrusts)

        for i, motor_name in enumerate(self.motor_names):
            nn = self.motor_dyn_nets[motor_name].double()
            omega_tensor = torch.tensor(np.array(self.omega_buffer), dtype=torch.double)
            cthrust_tensor = torch.tensor(np.array(self.cthrust_buffer), dtype=torch.double)

            inp = torch.cat([omega_tensor[:, i], cthrust_tensor[:, i]])
            actual_thrusts[i] = nn(inp).item()

        return actual_thrusts
    def map_action_to_physical(self,action):
    

        MAX_THRUST_PER_MOTOR = 0.1573  # N
        MAX_TORQUE_PER_MOTOR = 3.842e-3  # Nm

        MAX_THRUST = 4 * MAX_THRUST_PER_MOTOR  # 0.6292 N
        MAX_MOMENT = 2 * MAX_TORQUE_PER_MOTOR  # conservative

        HOVER_THRUST = MASS * GRAVITY  # approximately 0.3236 N

        # Map action in [-1, 1] to physical control outputs.
        #thrust = ((action[0] + 1) / 2) * (MAX_THRUST - HOVER_THRUST) + HOVER_THRUST
        thrust = ((action[0] + 1) / 2) * (MAX_THRUST-0.26487)+0.26487
        mx = action[1] * MAX_MOMENT * TORQUE_SCALE
        my = action[2] * MAX_MOMENT * TORQUE_SCALE
        mz = action[3] * MAX_MOMENT * TORQUE_SCALE

        return thrust, mx, my, mz

    def set_motor_thrusts(self,motor_thrust,Mx,My,Mz):
        
        motor_speed = self.mixer.calculate(motor_thrust, Mx,My,Mz)
        self.data.actuator('motor1').ctrl[0] = calc_motor_input(motor_speed[0])
        self.data.actuator('motor2').ctrl[0] = calc_motor_input(motor_speed[1])
        self.data.actuator('motor3').ctrl[0] = calc_motor_input(motor_speed[2])
        self.data.actuator('motor4').ctrl[0] = calc_motor_input(motor_speed[3])

    def set_motor_thrusts_2(self,action):
        
        motor_speeds = (action + 1) / 2 * 22 
            
        for i, speed in enumerate(motor_speeds, 1):
          self.data.actuator(f'motor{i}').ctrl[0] = calc_motor_input(speed)
        
        


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


    def sim_dt(self):
        return self.model.opt.timestep

    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)


    def get_qpos(self):
        return self.data.qpos.copy()

    def get_qvel(self):
        return self.data.qvel.copy()

    def get_qacc(self):
        return self.data.qacc.copy()

    def get_cvel(self):
        return self.data.cvel.copy()

    def get_jnt_id_by_name(self, name):
        return self.model.joint(name)

    def get_jnt_qposadr_by_name(self, name):
        return self.model.joint(name).qposadr

    def get_jnt_qveladr_by_name(self, name):
        return self.model.joint(name).dofadr

    def get_body_ext_force(self):
        return self.data.cfrc_ext.copy()

    def get_motor_speed_limits(self):
        """
        Returns speed limits of the *actuator* in radians per sec.
        This assumes the actuator 'user' element defines speed limits
        at the actuator level in revolutions per minute.
        """
        rpm_limits = self.model.actuator_user[:,0] # RPM
        return ((rpm_limits)*(2*np.pi/60)) # radians per sec

    def get_act_joint_speed_limits(self):
        """
        Returns speed limits of the *joint* in radians per sec.
        This assumes the actuator 'user' element defines speed limits
        at the actuator level in revolutions per minute.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        mot_lims = self.get_motor_speed_limits()
        return [float(i/j) for i,j in zip(mot_lims, gear_ratios)]

    def get_gear_ratios(self):
        """
        Returns transmission ratios.
        """
        return self.model.actuator_gear[:,0]

    def get_motor_names(self):
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return actuator_names

    def get_actuated_joint_inds(self):
        """
        Returns list of joint indices to which actuators are attached.
        """
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return [idx for idx, jnt in enumerate(joint_names) if jnt+'_motor' in actuator_names]

    def get_actuated_joint_names(self):
        """
        Returns list of joint names to which actuators are attached.
        """
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return [jnt for idx, jnt in enumerate(joint_names) if jnt+'_motor' in actuator_names]

    def get_motor_qposadr(self):
        """
        Returns the list of qpos indices of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        return [self.model.jnt_qposadr[i] for i in indices]

    def get_motor_positions(self):
        """
        Returns position of actuators.
        """
        return self.data.actuator_length

    def get_motor_velocities(self):
        """
        Returns velocities of actuators.
        """
        return self.data.actuator_velocity

    def get_act_joint_torques(self):
        """
        Returns actuator force in joint space.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_torques = self.data.actuator_force
        return (motor_torques*gear_ratios)

    def get_act_joint_positions(self):
        """
        Returns position of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_positions = self.get_motor_positions()
        return (motor_positions/gear_ratios)

    def get_act_joint_velocities(self):
        """
        Returns velocities of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_velocities = self.get_motor_velocities()
        return (motor_velocities/gear_ratios)

    def get_act_joint_position(self, act_name):
        """
        Returns position of actuator at joint level.
        """
        assert len(self.data.actuator(act_name).length)==1
        return self.data.actuator(act_name).length[0]/self.model.actuator(act_name).gear[0]

    def get_act_joint_velocity(self, act_name):
        """
        Returns velocity of actuator at joint level.
        """
        assert len(self.data.actuator(act_name).velocity)==1
        return self.data.actuator(act_name).velocity[0]/self.model.actuator(act_name).gear[0]

    def get_act_joint_ranges(self):
        """
        Returns the lower and upper limits of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        low, high = self.model.jnt_range[indices, :].T
        return low, high

    def get_act_joint_range(self, act_name):
        """
        Returns the lower and upper limits of given joint.
        """
        low, high = self.model.joint(act_name).range
        return low, high

    def get_actuator_ctrl_range(self):
        """
        Returns the acutator ctrlrange defined in model xml.
        """
        low, high = self.model.actuator_ctrlrange.copy().T
        return low, high

    def get_actuator_user_data(self):
        """
        Returns the user data (if any) attached to each actuator.
        """
        return self.model.actuator_user.copy()

    def get_root_body_pos(self):
        return self.data.xpos[1].copy()

    def get_root_body_vel(self):
        qveladr = self.get_jnt_qveladr_by_name("root")
        return self.data.qvel[qveladr:qveladr+6].copy()

    def get_sensordata(self, sensor_name):
        sensor = self.model.sensor(sensor_name)
        sensor_adr = sensor.adr[0]
        data_dim = sensor.dim[0]
        return self.data.sensordata[sensor_adr:sensor_adr+data_dim]

    def get_rfoot_body_pos(self):
        if isinstance(self.rfoot_body_name, list):
            return [self.data.body(i).xpos.copy() for i in self.rfoot_body_name]
        return self.data.body(self.rfoot_body_name).xpos.copy()

    def get_lfoot_body_pos(self):
        if isinstance(self.lfoot_body_name, list):
            return [self.data.body(i).xpos.copy() for i in self.lfoot_body_name]
        return self.data.body(self.lfoot_body_name).xpos.copy()

    def get_body_floor_contacts(self, body_name):
        """
        Returns list of 'body' and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        body_contacts = []

        body_names = [body_name] if isinstance(body_name, str) else body_name
        body_ids = [self.model.body(bn).id for bn in body_names]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_body = (self.model.geom_bodyid[c.geom2] in body_ids)
            if (geom1_is_floor and geom2_is_body):
                body_contacts.append((i,c))
        return body_contacts

    def get_rfoot_floor_contacts(self):
        """
        Returns list of right foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        rcontacts = []

        rfeet = [self.rfoot_body_name] if isinstance(self.rfoot_body_name, str) else self.rfoot_body_name
        rfeet_ids = [self.model.body(bn).id for bn in rfeet]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_rfoot = (self.model.geom_bodyid[c.geom2] in rfeet_ids)
            if (geom1_is_floor and geom2_is_rfoot):
                rcontacts.append((i,c))
        return rcontacts

    def get_lfoot_floor_contacts(self):
        """
        Returns list of left foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        lcontacts = []

        lfeet = [self.lfoot_body_name] if isinstance(self.lfoot_body_name, str) else self.lfoot_body_name
        lfeet_ids = [self.model.body(bn).id for bn in lfeet]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_lfoot = (self.model.geom_bodyid[c.geom2] in lfeet_ids)
            if (geom1_is_floor and geom2_is_lfoot):
                lcontacts.append((i,c))
        return lcontacts

    def get_rfoot_grf(self):
        """
        Returns total Ground Reaction Force on right foot.
        """
        right_contacts = self.get_rfoot_floor_contacts()
        rfoot_grf = 0
        for i, con in right_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            rfoot_grf += np.linalg.norm(c_array)
        return rfoot_grf

    def get_lfoot_grf(self):
        """
        Returns total Ground Reaction Force on left foot.
        """
        left_contacts = self.get_lfoot_floor_contacts()
        lfoot_grf = 0
        for i, con in left_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            lfoot_grf += (np.linalg.norm(c_array))
        return lfoot_grf

    def get_body_contact_force(self, body):
        """
        Returns total contact force acting on a body (or list of bodies).
        """
        if isinstance(body, str):
            body = [body]
        frc = 0
        for i, con in enumerate(self.data.contact):
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            b1 = self.model.body(self.model.geom(con.geom1).bodyid)
            b2 = self.model.body(self.model.geom(con.geom2).bodyid)
            if b1.name in body or b2.name in body:
                frc += np.linalg.norm(c_array)
        return frc

    def get_interaction_force(self, body1, body2):
        """
        Returns contact force beween a body1 and body2.
        """
        frc = 0
        for i, con in enumerate(self.data.contact):
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            b1 = self.model.body(self.model.geom(con.geom1).bodyid)
            b2 = self.model.body(self.model.geom(con.geom2).bodyid)
            if (b1.name==body1 and b2.name==body2) or (b1.name==body2 and b2.name==body1):
                frc += np.linalg.norm(c_array)
        return frc

    def get_body_vel(self, body_name, frame=0):
        """
        Returns translational and rotational velocity of a body in body-centered frame, world/local orientation.
        """
        body_vel = np.zeros(6)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 body_id, body_vel, frame)
        return [body_vel[3:6], body_vel[0:3]]

    def get_rfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of right foot.
        """
        if isinstance(self.rfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.rfoot_body_name]
        return self.get_body_vel(self.rfoot_body_name, frame=frame)

    def get_lfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of left foot.
        """
        if isinstance(self.lfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.lfoot_body_name]
        return self.get_body_vel(self.lfoot_body_name, frame=frame)

    def get_object_xpos_by_name(self, object_name, object_type):
        if object_type=="OBJ_BODY":
            return self.data.body(object_name).xpos
        elif object_type=="OBJ_GEOM":
            return self.data.geom(object_name).xpos
        elif object_type=="OBJ_SITE":
            return self.data.site(object_name).xpos
        else:
            raise Exception("object type should either be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")

    def get_object_xquat_by_name(self, object_name, object_type):
        if object_type=="OBJ_BODY":
            return self.data.body(object_name).xquat
        if object_type=="OBJ_GEOM":
            xmat = self.data.geom(object_name).xmat
            return tf3.quaternions.mat2quat(xmat)
        if object_type=="OBJ_SITE":
            xmat = self.data.site(object_name).xmat
            return tf3.quaternions.mat2quat(xmat)
        else:
            raise Exception("object type should be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")

    def get_object_affine_by_name(self, object_name, object_type):
        """Helper to create transformation matrix from position and quaternion."""
        pos = self.get_object_xpos_by_name(object_name, object_type)
        quat = self.get_object_xquat_by_name(object_name, object_type)
        return tf3.affines.compose(pos, tf3.quaternions.quat2mat(quat), np.ones(3))

    def get_robot_com(self):
        """
        Returns the center of mass of subtree originating at root body
        i.e. the CoM of the entire robot body in world coordinates.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreecom' not in sensor_names:
            raise Exception("subtree_com sensor not attached.")
        return self.data.subtree_com[1].copy()

    def get_robot_linmom(self):
        """
        Returns linear momentum of robot in world coordinates.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreelinvel' not in sensor_names:
            raise Exception("subtree_linvel sensor not attached.")
        linvel = self.data.subtree_linvel[1].copy()
        total_mass = self.get_robot_mass()
        return linvel*total_mass

    def get_robot_angmom(self):
        """
        Return angular momentum of robot's CoM about the world origin.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreeangmom' not in sensor_names:
            raise Exception("subtree_angmom sensor not attached.")
        com_pos = self.get_robot_com()
        lin_mom = self.get_robot_linmom()
        return self.data.subtree_angmom[1] + np.cross(com_pos, lin_mom)


    def check_rfoot_floor_collision(self):
        """
        Returns True if there is a collision between right foot and floor.
        """
        return (len(self.get_rfoot_floor_contacts())>0)

    def check_lfoot_floor_collision(self):
        """
        Returns True if there is a collision between left foot and floor.
        """
        return (len(self.get_lfoot_floor_contacts())>0)

    def check_bad_collisions(self, body_names=[]):
        """
        Returns True if there are collisions other than specifiedbody-floor,
        or feet-floor if body_names is not provided.
        """
        num_cons = 0
        if not isinstance(body_names, list):
            raise TypeError(f"expected list of body names, got '{type(body_names).__name__}'")
        if not len(body_names):
            num_rcons = len(self.get_rfoot_floor_contacts())
            num_lcons = len(self.get_lfoot_floor_contacts())
            num_cons = num_rcons + num_lcons
        for bn in body_names:
            num_cons += len(self.get_body_floor_contacts(bn))
        return num_cons != self.data.ncon

    def check_self_collisions(self):
        """
        Returns True if there are collisions other than any-geom-floor.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_robot = self.model.body(geom1_body.rootid).name==self.robot_root_name
            geom2_is_robot = self.model.body(geom2_body.rootid).name==self.robot_root_name
            if geom1_is_robot and geom2_is_robot:
                return True
        return False

    def set_pd_gains(self, kp, kv):
        assert kp.size==self.model.nu
        assert kv.size==self.model.nu
        self.kp = kp.copy()
        self.kv = kv.copy()
        return

    def step_pd(self, p, v):
        target_angles = p
        target_speeds = v

        assert type(target_angles)==np.ndarray
        assert type(target_speeds)==np.ndarray

        curr_angles = self.get_act_joint_positions()
        curr_speeds = self.get_act_joint_velocities()

        perror = target_angles - curr_angles
        verror = target_speeds - curr_speeds

        assert self.kp.size==perror.size
        assert self.kv.size==verror.size
        return self.kp * perror + self.kv * verror

    def set_motor_torque(self, torque, motor_dyn_fwd = False):
        """
        Apply torques to motors.
        """
        if isinstance(torque, np.ndarray):
            assert torque.shape==(self.nu(), )
            ctrl = torque
        elif isinstance(torque, list):
            assert len(torque)==self.nu()
            ctrl = np.copy(torque)
        else:
            raise Exception("motor torque should be list of ndarray.")
        try:
            if motor_dyn_fwd:
                if not hasattr(self, 'motor_dyn_nets'):
                    raise Exception("motor dynamics network are not defined.")
                gear = self.get_gear_ratios()
                ctrl = self.motor_nets_forward(ctrl*gear)
                ctrl /= gear
            np.copyto(self.data.ctrl, ctrl)
        except Exception as e:
            print("Could not apply motor torque.")
            print(e)
        return

    def step(self, mj_step=True, nstep=1):
        """
        (Adapted from dm_control/mujoco/engine.py)

        Advances physics with up-to-date position and velocity dependent fields.
        Args:
          nstep: Optional integer, number of steps to take.
        """
        if mj_step:
            mujoco.mj_step(self.model, self.data, nstep)
            self.step_counter += nstep
            return

        # In the case of Euler integration we assume mj_step1 has already been
        # called for this state, finish the step with mj_step2 and then update all
        # position and velocity related fields with mj_step1. This ensures that
        # (most of) mjData is in sync with qpos and qvel. In the case of non-Euler
        # integrators (e.g. RK4) an additional mj_step1 must be called after the
        # last mj_step to ensure mjData syncing.
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4.value:
          mujoco.mj_step2(self.model, self.data)
          if nstep > 1:
            mujoco.mj_step(self.model, self.data, nstep-1)
        else:
          mujoco.mj_step(self.model, self.data, nstep)

        mujoco.mj_step1(self.model, self.data)

        self.step_counter += nstep
