"""
Defensive obstacle avoidance controller.

When the robot's lidar reports an obstacle within a threshold distance,
this controller takes over and steers toward the direction of maximum
clearance (the lidar ray with the longest free range). When all relevant
lidar rays are clear, control returns to the underlying policy (TD3+LQR).

This is the "tumbler" behavior the supervisor described in the meeting:
multiple scenarios with hard switching by error delta (here: by min lidar).

The controller produces (left_wheel, right_wheel) action in the same
normalized [-1, 1] space as TD3, so it can be directly substituted for
the policy output when in AVOIDANCE mode.

References:
- The supervisor's request for scenario switching (2026-05-08 meeting,
  blocks B and C in the journal).
- Classical bug-style obstacle avoidance from mobile robotics
  (Borenstein & Koren, 1991; Ulrich & Borenstein VFH, 1998 -- conceptually
  similar but simpler).

Usage:
    avoider = ObstacleAvoidanceController()
    if avoider.should_take_over(obs_lidar):
        action = avoider.compute_action(obs_lidar, robot_yaw, goal_dx, goal_dy)
    else:
        action = td3_action  # or td3 + lqr residual
"""

from __future__ import annotations

import math

import numpy as np


# Default thresholds. Override at construction if needed.
DEFAULT_TAKEOVER_DIST = 1.5       # meters: takeover if any front ray < this
DEFAULT_RELEASE_DIST = 2.5        # meters: release if all front rays > this (hysteresis)
DEFAULT_FRONT_HALF_WIDTH = 6      # rays on each side of forward (wider arc)
DEFAULT_FORWARD_SPEED = 0.35      # normalized [0, 1] forward speed during avoidance
DEFAULT_TURN_GAIN = 1.5           # how aggressively to turn toward clearance


class ObstacleAvoidanceController:
    """Heuristic avoidance with hysteresis between takeover and release.

    Takeover (entering avoidance mode):
        if any of the front rays is shorter than `takeover_dist`.

    Release (exiting back to underlying policy):
        if ALL front rays are longer than `release_dist`.

    Steering during avoidance:
        find the lidar ray with the maximum clearance among ALL rays;
        compute angle from forward to that ray;
        turn toward it with magnitude proportional to that angle;
        forward speed is fixed at `forward_speed` (slowed-down crawl).

    The state is kept inside the instance so the hysteresis works across
    a whole episode. Reset by calling `reset()` between episodes.
    """

    def __init__(
        self,
        n_lidar_rays: int = 16,
        takeover_dist: float = DEFAULT_TAKEOVER_DIST,
        release_dist: float = DEFAULT_RELEASE_DIST,
        front_half_width: int = DEFAULT_FRONT_HALF_WIDTH,
        forward_speed: float = DEFAULT_FORWARD_SPEED,
        turn_gain: float = DEFAULT_TURN_GAIN,
    ):
        self.n_lidar_rays = int(n_lidar_rays)
        self.takeover_dist = float(takeover_dist)
        self.release_dist = float(release_dist)
        self.front_half_width = int(front_half_width)
        self.forward_speed = float(forward_speed)
        self.turn_gain = float(turn_gain)
        self._in_avoidance = False

    def reset(self) -> None:
        self._in_avoidance = False

    @property
    def is_active(self) -> bool:
        return self._in_avoidance

    def _front_ray_indices(self) -> list[int]:
        """Indices of rays considered 'in front' of the robot.

        Lidar layout in HuskyObstacleEnv: ray i has angle = yaw + 2*pi*i/N,
        so ray 0 points along the robot's heading (forward). Front rays
        are then 0, 1, ..., front_half_width and (N-1, N-2, ...,
        N-front_half_width).
        """
        idx = list(range(0, self.front_half_width + 1))
        idx += list(range(self.n_lidar_rays - self.front_half_width,
                          self.n_lidar_rays))
        return sorted(set(idx))

    def should_take_over(self, lidar: np.ndarray) -> bool:
        """Decide whether avoidance mode should be active for this step.

        Implements hysteresis:
          - Takeover: any FRONT ray < takeover_dist.
          - Release: ALL rays (front + sides, NOT just front) >
            release_dist. This ensures we don't release while an
            obstacle is still alongside the robot, where it could
            become a front-ray collision again on the next yaw change.
        """
        lidar_arr = np.asarray(lidar)
        front = lidar_arr[self._front_ray_indices()]

        # All rays except the strictly-rear ones:
        # rays 0..N/2 (front + right + part of right-back) and
        # N/2..N (left + part of left-back).
        # Simpler: take ALL rays for release check -- the sides matter too.
        all_relevant = lidar_arr  # full 360 degree view

        if not self._in_avoidance:
            if float(front.min()) < self.takeover_dist:
                self._in_avoidance = True
        else:
            # Stay in avoidance until NOTHING is close (front or side)
            if float(all_relevant.min()) > self.release_dist:
                self._in_avoidance = False

        return self._in_avoidance

    def compute_action(self, lidar: np.ndarray) -> np.ndarray:
        """Compute (left_wheel, right_wheel) action that steers AWAY from
        the closest obstacle direction.

        Strategy: rather than "go where it's clear" (which can underturn
        and send the robot grazing into the obstacle), we identify the
        direction of the closest obstacle in the front hemisphere and
        turn by 90 degrees away from it. This guarantees a strong
        evasive turn proportional to obstacle proximity.

        If the closest obstacle is directly ahead (angle ~ 0), we pick
        a side based on which side has more total clearance.
        """
        lidar = np.asarray(lidar, dtype=np.float32)

        angles = np.array(
            [2.0 * math.pi * i / self.n_lidar_rays
             for i in range(self.n_lidar_rays)],
            dtype=np.float32,
        )
        angles = np.where(angles > math.pi, angles - 2.0 * math.pi, angles)

        # Consider only front hemisphere for finding the obstacle direction
        front_mask = np.abs(angles) <= (math.pi / 2.0 + 0.1)
        # For finding closest obstacle, use lidar values; for non-front rays
        # set to a large value so they don't win the argmin.
        front_lidar_for_min = np.where(front_mask, lidar, 1e6)
        closest_idx = int(np.argmin(front_lidar_for_min))
        obstacle_angle = float(angles[closest_idx])

        # Turn 90 degrees AWAY from the obstacle.
        # If obstacle is at angle a (in [-pi, pi]), turn target is a +/- pi/2,
        # whichever lies in the front hemisphere.
        # Sign convention: positive turn = turn left (positive yaw rate).
        # Obstacle on the right (a < 0) -> turn left (target = a + pi/2 > 0).
        # Obstacle on the left  (a > 0) -> turn right (target = a - pi/2 < 0).
        # Obstacle dead ahead (a ~ 0)   -> pick side by total clearance.
        if abs(obstacle_angle) < 1e-3:
            # Tie-break by total clearance on each side
            n_quarter = self.n_lidar_rays // 4
            left_clr = float(lidar[1: n_quarter + 1].mean())
            right_clr = float(lidar[self.n_lidar_rays - n_quarter:].mean())
            if left_clr >= right_clr:
                target_angle = +math.pi / 2.0
            else:
                target_angle = -math.pi / 2.0
        elif obstacle_angle > 0:
            # Obstacle on the left -> evade right
            target_angle = obstacle_angle - math.pi / 2.0
        else:
            # Obstacle on the right -> evade left
            target_angle = obstacle_angle + math.pi / 2.0

        # Differential-drive command: forward + turn
        # turn > 0 means turn left (positive angular velocity)
        turn = float(np.clip(self.turn_gain * target_angle / math.pi, -1.0, 1.0))

        forward = self.forward_speed

        # Differential mapping: left wheel = forward - turn, right = forward + turn
        # (turn left => slow left wheel, speed up right wheel)
        left = forward - turn
        right = forward + turn

        action = np.array([left, right], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)
