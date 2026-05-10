"""
Goal redirection: cosmetic obstacle avoidance via observation hijacking.

Idea: the underlying TD3 policy was trained on observations
[..., goal_dx, goal_dy] in world frame. If we replace these two values
with a "virtual goal" that lies SIDE-WAYS of the obstacle, the policy
will steer toward the virtual point, walking around the obstacle. Once
the obstacle is no longer in the way, we restore the true goal.

The TD3 policy itself is unchanged. Only the observation it sees is
modified. The reward, termination, and physics all use the true goal.

This is conceptually a one-step waypoint that's automatically placed
based on the lidar reading, rather than a static geometric subdivision
of the path (which is what WaypointPlanner does and which doesn't help
when an obstacle is on the line).

When to use this vs ObstacleAvoidanceController:
- Goal redirect: smooth trajectory, uses the trained policy's behavior,
  preserves LQR residual contribution. Good when the policy has SOME
  ability to maneuver but just doesn't see the obstacle as relevant
  to the current goal direction.
- ObstacleAvoidanceController: hard takeover, ignores policy entirely
  during avoidance. Use when policy is hopeless and we need a
  guaranteed avoidance behavior.

Both can be combined: redirect first, fall back to hard avoidance if
the lidar gets too close anyway.
"""

from __future__ import annotations

import math

import numpy as np


DEFAULT_REDIRECT_DIST = 2.5       # meters: redirect goal if any front ray < this
DEFAULT_DETOUR_DIST = 4.0         # meters: how far to the side to place virtual goal
DEFAULT_FRONT_HALF_WIDTH = 6      # rays on each side of forward to consider


class GoalRedirector:
    """Hijack observation's goal vector to detour around obstacles.

    Each step (called before model.predict) we look at the front lidar
    rays. If anything is within REDIRECT_DIST, we generate a virtual goal
    placed perpendicular to the original goal direction at distance
    DETOUR_DIST, on the side that has more clearance.

    The replacement only affects observation indices 7 and 8
    (goal_dx, goal_dy in world frame). Everything else is untouched.
    """

    def __init__(
        self,
        n_lidar_rays: int = 16,
        redirect_dist: float = DEFAULT_REDIRECT_DIST,
        detour_dist: float = DEFAULT_DETOUR_DIST,
        front_half_width: int = DEFAULT_FRONT_HALF_WIDTH,
    ):
        self.n_lidar_rays = int(n_lidar_rays)
        self.redirect_dist = float(redirect_dist)
        self.detour_dist = float(detour_dist)
        self.front_half_width = int(front_half_width)
        self._active = False

    def reset(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def _front_ray_indices(self) -> list[int]:
        idx = list(range(0, self.front_half_width + 1))
        idx += list(range(self.n_lidar_rays - self.front_half_width,
                          self.n_lidar_rays))
        return sorted(set(idx))

    def maybe_redirect(self, obs: np.ndarray) -> np.ndarray:
        """Return a possibly-modified observation with hijacked goal vector.

        obs layout (HuskyObstacleEnv, 25-D):
            [0..6]   = pose, velocity, orientation
            [7]      = goal_dx (world frame)
            [8]      = goal_dy (world frame)
            [9..24]  = 16 lidar ranges

        Lidar ray i has world-frame angle yaw + 2*pi*i/N. So to figure out
        which side has more space, we use lidar values directly: rays in
        the LEFT-front quadrant are i in [1..N/4], rays in the RIGHT-front
        are [N - N/4 .. N-1]. Whichever side has greater mean clearance is
        the side we detour to.
        """
        obs = np.asarray(obs, dtype=np.float32).copy()
        lidar = obs[-self.n_lidar_rays:]
        front = lidar[self._front_ray_indices()]

        if float(front.min()) >= self.redirect_dist:
            self._active = False
            return obs

        # Need to redirect. Pick a side.
        n_quarter = self.n_lidar_rays // 4
        left_rays = lidar[1: n_quarter + 1]
        right_rays = lidar[self.n_lidar_rays - n_quarter: self.n_lidar_rays]
        left_clearance = float(left_rays.mean())
        right_clearance = float(right_rays.mean())

        # Direction of detour in robot's local frame:
        # +y is left, -y is right.
        if left_clearance >= right_clearance:
            detour_local_y = +self.detour_dist
        else:
            detour_local_y = -self.detour_dist

        # Convert detour from robot-local to world frame using yaw.
        cos_yaw, sin_yaw = float(obs[5]), float(obs[6])
        # Local frame: x = forward (along heading), y = left of heading.
        # Rotation by yaw: [cos -sin; sin cos] @ [local_x, local_y]
        # We want detour_local = (0, detour_local_y) -> world offset.
        detour_world_dx = -sin_yaw * detour_local_y
        detour_world_dy = +cos_yaw * detour_local_y

        # Replace goal_dx, goal_dy with virtual goal direction.
        # Virtual goal is current robot position + detour vector, so
        # (goal_dx, goal_dy) becomes just the detour vector itself.
        obs[7] = detour_world_dx
        obs[8] = detour_world_dy

        self._active = True
        return obs
