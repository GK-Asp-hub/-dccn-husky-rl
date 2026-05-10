"""
Map-aware obstacle avoidance for known obstacle positions.

Unlike ObstacleAvoidanceController which uses raw lidar (and can miss
obstacles between rays at 22.5-degree spacing), this controller uses
the known obstacle positions directly. This is appropriate when the
obstacle map is available (deterministic experiments, environments
with planning, supervisory map sharing).

Use this for Experiment A where the obstacle is fixed and known by
construction. For real-world adaptation, replace the obstacle list
input with a SLAM map or perception output.
"""

from __future__ import annotations

import math
import numpy as np


class MapAwareAvoider:
    """Avoidance controller that knows obstacle positions explicitly.

    Activation: any obstacle in the front hemisphere is closer than
    `takeover_dist` (measured robot center to obstacle edge).
    Release: all obstacles farther than `release_dist`.

    Action: turn 90 degrees away from the closest obstacle in the
    front hemisphere; move forward at `forward_speed`.
    """

    def __init__(
        self,
        takeover_dist: float = 1.5,
        release_dist: float = 2.5,
        forward_speed: float = 0.4,
        turn_gain: float = 1.5,
    ):
        self.takeover_dist = float(takeover_dist)
        self.release_dist = float(release_dist)
        self.forward_speed = float(forward_speed)
        self.turn_gain = float(turn_gain)
        self._in_avoidance = False

    def reset(self):
        self._in_avoidance = False

    @property
    def is_active(self) -> bool:
        return self._in_avoidance

    def _obstacle_distances_and_angles(
        self,
        robot_x: float, robot_y: float, robot_yaw: float,
        obstacles: list[tuple[float, float, float]],
    ):
        """For each obstacle, return (signed_distance_to_edge, angle_in_robot_frame).

        angle is in [-pi, pi], where 0 = forward, +pi/2 = left.
        """
        results = []
        for ox, oy, oradius in obstacles:
            dx = ox - robot_x
            dy = oy - robot_y
            dist_to_center = math.hypot(dx, dy)
            dist_to_edge = dist_to_center - oradius

            world_angle = math.atan2(dy, dx)
            local_angle = world_angle - robot_yaw
            # Wrap to [-pi, pi]
            while local_angle > math.pi:
                local_angle -= 2 * math.pi
            while local_angle < -math.pi:
                local_angle += 2 * math.pi
            results.append((dist_to_edge, local_angle))
        return results

    def should_take_over(self, robot_x, robot_y, robot_yaw, obstacles):
        info = self._obstacle_distances_and_angles(robot_x, robot_y, robot_yaw, obstacles)

        # Find minimum distance among FRONT-hemisphere obstacles (|angle| < pi/2 + margin)
        front_dists = [d for d, a in info if abs(a) < math.pi / 2.0 + 0.1]

        if not self._in_avoidance:
            if front_dists and min(front_dists) < self.takeover_dist:
                self._in_avoidance = True
        else:
            # Release only if ALL obstacles are far (front or side)
            all_dists = [d for d, _ in info]
            if not all_dists or min(all_dists) > self.release_dist:
                self._in_avoidance = False

        return self._in_avoidance

    def compute_action(self, robot_x, robot_y, robot_yaw, obstacles) -> np.ndarray:
        """Turn away from the closest obstacle, move forward slowly."""
        info = self._obstacle_distances_and_angles(robot_x, robot_y, robot_yaw, obstacles)
        # Closest obstacle (by edge distance), considering all of them
        closest = min(info, key=lambda x: x[0])
        dist_edge, obstacle_angle = closest

        # Evade target: 90 degrees away from obstacle direction.
        if abs(obstacle_angle) < 1e-3:
            # Obstacle dead ahead; pick a side. Default: turn left.
            target_angle = +math.pi / 2.0
        elif obstacle_angle > 0:
            target_angle = obstacle_angle - math.pi / 2.0
        else:
            target_angle = obstacle_angle + math.pi / 2.0

        turn = float(np.clip(self.turn_gain * target_angle / math.pi, -1.0, 1.0))
        forward = self.forward_speed
        left = forward - turn
        right = forward + turn
        return np.clip(np.array([left, right], dtype=np.float32), -1.0, 1.0)
