# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import numpy as np
import torch


# ==============================
# Wrap normal env
# ==============================
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        state, reward, done, info = self.env.step(action[0])
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])

    def reset(self):
        return np.array([self.env.reset()])


# ==============================
#  Mirror Symmetry Wrapper
# ==============================
class SymmetricEnv:
    def __init__(self, env_fn,
                 mirrored_obs,
                 mirrored_act):

        if mirrored_obs:
            self.obs_mirror_matrix = torch.Tensor(
                _get_symmetry_matrix(mirrored_obs)
            )

        if mirrored_act:
            self.act_mirror_matrix = torch.Tensor(
                _get_symmetry_matrix(mirrored_act)
            )

        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    # ---- Mirror Action ----
    def mirror_action(self, action):
        """
        action: shape (batch, 3)
        """
        return action @ self.act_mirror_matrix

    # ---- Mirror Observation ----
    def mirror_observation(self, obs):
        """
        obs shape: (batch, obs_dim)
        """
        return obs @ self.obs_mirror_matrix


# ==============================
# symmetry matrix builder
# e.g. mirrored = [-1, 1, 1]
# ==============================
def _get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    mat = np.zeros((numel, numel))

    for i, j in zip(np.arange(numel), np.abs(np.array(mirrored).astype(int))):
        mat[i, j] = np.sign(mirrored[i])

    return mat
