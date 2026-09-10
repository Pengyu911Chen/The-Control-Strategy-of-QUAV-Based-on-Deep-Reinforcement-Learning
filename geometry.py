# ----------------------------------------------------------------------------
# Copyright 2026, Pengyu Chen
# Johns Hopkins University
# All Rights Reserved
# Authors: Pengyu Chen
# See LICENSE file for the license information
# ----------------------------------------------------------------------------

import numpy as np

class GeoQuaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def setFromAngleAxis(self, angle, axis):
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2.0
        sin_half_angle = np.sin(half_angle)
        self.w = np.cos(half_angle)
        self.x = axis[0] * sin_half_angle
        self.y = axis[1] * sin_half_angle
        self.z = axis[2] * sin_half_angle

    def setFromRotationMatrix(self, mat):
        trace = np.trace(mat)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            self.w = 0.25 / s
            self.x = (mat[2, 1] - mat[1, 2]) * s
            self.y = (mat[0, 2] - mat[2, 0]) * s
            self.z = (mat[1, 0] - mat[0, 1]) * s
        else:
            if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
                self.w = (mat[2, 1] - mat[1, 2]) / s
                self.x = 0.25 * s
                self.y = (mat[0, 1] + mat[1, 0]) / s
                self.z = (mat[0, 2] + mat[2, 0]) / s
            elif mat[1, 1] > mat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
                self.w = (mat[0, 2] - mat[2, 0]) / s
                self.x = (mat[0, 1] + mat[1, 0]) / s
                self.y = 0.25 * s
                self.z = (mat[1, 2] + mat[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
                self.w = (mat[1, 0] - mat[0, 1]) / s
                self.x = (mat[0, 2] + mat[2, 0]) / s
                self.y = (mat[1, 2] + mat[2, 1]) / s
                self.z = 0.25 * s

    def getRotationMatrix(self):
        x2 = self.x + self.x
        y2 = self.y + self.y
        z2 = self.z + self.z
        xx = self.x * x2
        yy = self.y * y2
        zz = self.z * z2
        xy = self.x * y2
        xz = self.x * z2
        yz = self.y * z2
        wx = self.w * x2
        wy = self.w * y2
        wz = self.w * z2

        m = np.zeros((3, 3))
        m[0, 0] = 1.0 - (yy + zz)
        m[0, 1] = xy - wz
        m[0, 2] = xz + wy
        m[1, 0] = xy + wz
        m[1, 1] = 1.0 - (xx + zz)
        m[1, 2] = yz - wx
        m[2, 0] = xz - wy
        m[2, 1] = yz + wx
        m[2, 2] = 1.0 - (xx + yy)

        return m

    def normalize(self):
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm == 0.0:
            return
        self.w /= norm
        self.x /= norm
        self.y /= norm
        self.z /= norm

    def slerp(self, t, other):
        cos_theta = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

        if cos_theta < 0.0:
            other = GeoQuaternion(-other.x, -other.y, -other.z, -other.w)
            cos_theta = -cos_theta

        if cos_theta > 0.9995:
            w = self.w + t * (other.w - self.w)
            x = self.x + t * (other.x - self.x)
            y = self.y + t * (other.y - self.y)
            z = self.z + t * (other.z - self.z)
            result = GeoQuaternion(x, y, z, w)
            result.normalize()
            return result

        theta = np.arccos(cos_theta)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        a = np.sin((1.0 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta

        w = a * self.w + b * other.w
        x = a * self.x + b * other.x
        y = a * self.y + b * other.y
        z = a * self.z + b * other.z
        return GeoQuaternion(x, y, z, w)

def hatmap(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def veemap(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]])

def skewSym(w):
    return hatmap(w)

def so3LieToMat(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)
    wx = hatmap(w / theta)
    return (
        np.eye(3)
        + np.sin(theta) * wx
        + (1.0 - np.cos(theta)) * (wx @ wx)
    )

def se3LieToRotTrans3(w, u):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        R = np.eye(3)
        t = u
    else:
        wx = hatmap(w / theta)
        R = so3LieToMat(w)
        A = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * wx
            + (theta - np.sin(theta)) / theta * (wx @ wx)
        )
        t = A @ u
    return R, t

def se3LieToMat4(w, u):
    R, t = se3LieToRotTrans3(w, u)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T
