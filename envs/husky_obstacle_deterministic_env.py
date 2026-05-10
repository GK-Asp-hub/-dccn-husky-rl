"""
HuskyObstacleDeterministicEnv -- detailed deterministic configuration for Experiment A.

Goal: demonstrate definitively "robot avoids obstacle and reaches goal".
All randomness removed:
- Goal at fixed point (default 4 m straight ahead).
- Obstacle exactly on the line between start (0,0) and goal.
- Obstacle radius and lateral offset configurable.

Uses HuskyObstacleEnv as base (lidar, collision termination, proximity penalty).
Only _spawn_obstacles and _sample_goal are overridden to make the layout deterministic.

Context: 2026-05-10 practice session, defense prep for 2026-05-12.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pybullet as p

from envs.husky_obstacle_env import (
    HuskyObstacleEnv,
    OBSTACLE_HEIGHT,
)


# --- Default deterministic layout ---
DEFAULT_GOAL_DISTANCE = 4.0          # goal at (GOAL_DISTANCE, 0)
DEFAULT_OBSTACLE_X = 2.0             # obstacle at midpoint along X
DEFAULT_OBSTACLE_Y_OFFSET = 0.0      # 0 = exactly on the line; >0 = lateral offset
DEFAULT_OBSTACLE_RADIUS = 0.5        # cylinder radius


class HuskyObstacleDeterministicEnv(HuskyObstacleEnv):
    """Husky drives toward a fixed goal with a single obstacle on the path.

    Inherits everything from HuskyObstacleEnv (lidar, collision, proximity penalty).
    Overrides only goal sampling and obstacle spawning to make them deterministic.

    Parameters
    ----------
    render_mode : str | None
        Passed to base PyBullet env ("human" / "rgb_array" / None).
    goal_distance : float
        X-coordinate of the goal. Robot starts at (0, 0).
    obstacle_x : float
        X-coordinate of the obstacle center.
    obstacle_y_offset : float
        Y-coordinate of the obstacle center. 0 = exactly on the line.
    obstacle_radius : float
        Radius of the cylindrical obstacle.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        goal_distance: float = DEFAULT_GOAL_DISTANCE,
        obstacle_x: float = DEFAULT_OBSTACLE_X,
        obstacle_y_offset: float = DEFAULT_OBSTACLE_Y_OFFSET,
        obstacle_radius: float = DEFAULT_OBSTACLE_RADIUS,
    ):
        self._goal_distance_param = float(goal_distance)
        self._obstacle_x_param = float(obstacle_x)
        self._obstacle_y_offset_param = float(obstacle_y_offset)
        self._obstacle_radius_param = float(obstacle_radius)
        super().__init__(render_mode=render_mode)

    # ---------- Deterministic goal ----------

    def _sample_goal(self) -> np.ndarray:
        """Always return the same goal point."""
        return np.array(
            [self._goal_distance_param, 0.0],
            dtype=np.float32,
        )

    # ---------- Deterministic obstacle ----------

    def _spawn_obstacles(self) -> None:
        """Spawn a single cylindrical obstacle at the configured position."""
        self._obstacle_body_ids = []
        self._obstacle_positions = []

        x = self._obstacle_x_param
        y = self._obstacle_y_offset_param
        r = self._obstacle_radius_param

        col = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER, radius=r, height=OBSTACLE_HEIGHT,
        )
        vis = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=r, length=OBSTACLE_HEIGHT,
            rgbaColor=[0.5, 0.35, 0.25, 1.0],
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, OBSTACLE_HEIGHT / 2],
        )
        self._obstacle_body_ids.append(body_id)
        self._obstacle_positions.append((x, y, r))
