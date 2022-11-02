#!/usr/bin/env python
# ---------------------------------------------------
import numpy as np
from geometry_msgs.msg import Point


class Transform:

    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z):
        self.workspace_x_min = min_x
        self.workspace_x_max = max_x

        self.workspace_y_min = min_y
        self.workspace_y_max = max_y

        self.workspace_z_min = min_z
        self.workspace_z_max = max_z

    def normalize_roll(self, denormal_roll):
        normalized = denormal_roll / np.pi  # roll is in range radian(-180) to radian(+ 180),
        # now it is in range -1 to +1
        return normalized

    def denormalize_roll(self,normal_roll):  # de-normalizing the roll
        denormalized = normal_roll * np.pi
        return denormalized

    def normalize_pitch(self,denormal_pitch):
        normalized = denormal_pitch / np.pi  # pitch is in range radian(-180) to radian(+ 180),
        # now it is in range -1 to +1
        return normalized

    def denormalize_pitch(self, normal_pitch):  # de-normalizing the pitch
        denormalized = normal_pitch * np.pi
        return denormalized

    def normalize_yaw(self, denormal_yaw):
        normalized = (denormal_yaw / np.pi) - 1
        return normalized  # yaw is in range radian(0) to radian(360), now it is in range -1 to +1

    def denormalizing_yaw(self, normal_yaw):
        denormalized = (normal_yaw + 1) * np.pi
        return denormalized

    def normalize_position(self, denormal_point):  # normalizing position to become in range -1 to 1, this is applicable
        # on x, y and z
        denormal_x = denormal_point.x
        denormal_y = denormal_point.y
        denormal_z = denormal_point.z
        normal_point = Point()
        normal_point.x = (denormal_x - self.workspace_x_min) / (self.workspace_x_max - self.workspace_x_min)
        normal_point.y = (denormal_y - self.workspace_y_min) / (self.workspace_y_max - self.workspace_y_min)
        normal_point.z = (denormal_z - self.workspace_z_min) / (self.workspace_z_max - self.workspace_z_min)
        return normal_point

    def normalize_epsilon(self, alpha):
        epsilon = Point()
        epsilon.x = alpha / (self.workspace_x_max - self.workspace_x_min)
        epsilon.y = alpha / (self.workspace_y_max - self.workspace_y_min)
        epsilon.z = alpha / (self.workspace_z_max - self.workspace_z_min)
        return epsilon

