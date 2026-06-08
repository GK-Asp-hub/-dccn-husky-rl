"""
HuskyMultiObstacleEnv -- детерминированная среда с произвольным списком препятствий
и произвольной точкой цели.

Обобщает HuskyObstacleDeterministicEnv (одно препятствие на прямой) на любые сценарии:
  - коридор из препятствий
  - барьер с проходом
  - S-образный слалом
  - случайное направление цели с несколькими детерминированными препятствиями

Робот по-прежнему стартует в (0, 0); цель можно поставить в любую точку арены.
Препятствия задаются списком (x, y, radius).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pybullet as p

from envs.husky_obstacle_env import (
    HuskyObstacleEnv,
    OBSTACLE_HEIGHT,
)


# Разумные дефолты — одно препятствие на прямой, как в исходном HuskyObstacleDeterministicEnv.
DEFAULT_GOAL = (4.0, 0.0)
DEFAULT_OBSTACLES = ((2.0, 0.0, 0.5),)


class HuskyMultiObstacleEnv(HuskyObstacleEnv):
    """Husky едет к фиксированной цели через произвольные фиксированные препятствия.

    Параметры
    ---------
    render_mode : str | None
        "human" / None, передаётся в базовую PyBullet-среду.
    goal : (gx, gy)
        Координаты цели в мировой СК. Робот стартует в (0, 0).
    obstacles : список (x, y, radius)
        Цилиндрические препятствия. Пустой список = пустая арена (лидар 16D всё равно
        выдаётся, но столкновений нет).
    """

    def __init__(
        self,
        render_mode: str | None = None,
        goal: tuple[float, float] = DEFAULT_GOAL,
        obstacles: Sequence[tuple[float, float, float]] = DEFAULT_OBSTACLES,
    ):
        self._goal_param = (float(goal[0]), float(goal[1]))
        self._obstacles_param = [
            (float(x), float(y), float(r)) for (x, y, r) in obstacles
        ]
        super().__init__(render_mode=render_mode)

    # ---------- Deterministic goal ----------

    def _sample_goal(self) -> np.ndarray:
        return np.array(self._goal_param, dtype=np.float32)

    # ---------- Deterministic obstacles ----------

    def _spawn_obstacles(self) -> None:
        self._obstacle_body_ids = []
        self._obstacle_positions = []

        for (x, y, r) in self._obstacles_param:
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


# ---------------- Predefined scenarios ----------------
# Each scenario is a (goal, obstacles) tuple ready to pass to the env.

SCENARIOS = {
    "single_on_line": (
        (4.0, 0.0),
        [(2.0, 0.0, 0.5)],
    ),
    "two_offset": (
        # Goal slightly to the right; two obstacles staggered around the path
        (4.5, 0.0),
        [(1.8, +0.6, 0.4), (3.0, -0.5, 0.4)],
    ),
    "three_corridor": (
        # Goal at 5m, three obstacles forming a corridor with narrow gaps
        (5.0, 0.0),
        [(1.5, +0.8, 0.4), (2.8, -0.7, 0.4), (4.0, +0.7, 0.4)],
    ),
    "barrier_with_gap": (
        # Three obstacles in a row at x=2.5, with a gap at y=0
        (4.5, 0.0),
        [(2.5, +1.2, 0.5), (2.5, -1.2, 0.5)],
    ),
    "slalom": (
        # Four obstacles forcing a snake path
        (5.5, 0.0),
        [(1.5, -0.6, 0.4), (2.7, +0.6, 0.4),
         (3.9, -0.6, 0.4), (5.1, +0.6, 0.4)],
    ),
    "diagonal_goal": (
        # Goal off-axis, one obstacle blocking direct path
        (3.5, 2.5),
        [(1.8, 1.3, 0.5)],
    ),
    "wide_obstacle": (
        # Single big obstacle, harder to go around
        (4.5, 0.0),
        [(2.2, 0.0, 0.9)],
    ),
}
