"""
HuskyGoalPlannedEnv — обёртка, подменяющая goal_vector в observation
на вектор до текущего waypoint'а планировщика.

Идея: базовый TD3 уже умеет ехать к произвольной точке, которую он видит
в obs[7:9] (goal_dx, goal_dy). Если вместо финальной цели показать ему
ближайший waypoint — он поедет туда. Когда робот достигнет waypoint'а,
планировщик переключится на следующий, и policy увидит новый goal_vector.

Важно: reward и terminated считаются БАЗОВЫМ env'ом по ФИНАЛЬНОЙ цели.
Планировщик влияет только на то, что видит policy — ни на награду, ни на
условия окончания эпизода.

Обёртка работает с любым env, удовлетворяющим контракту:
- observation[0:2] = (x, y) позиция робота
- observation[7:9] = (goal_dx, goal_dy) вектор к цели в мировой СК
- info из reset() содержит ключ "goal" с финальной целью
- первые 9 элементов obs остаются на своих местах независимо от остатка

Этому контракту удовлетворяют HuskyGoalEnv (Stage 1) и HuskyObstacleEnv
(Stage 2a, наследуется от HuskyGoalEnv и дополняет obs лидаром).

Контекст: 50_Journal/2026-04-21_planner_decision.md.
"""

from __future__ import annotations

from typing import Any, Type

import gymnasium as gym
import numpy as np

from envs.husky_goal_env import HuskyGoalEnv
from planners.waypoint_planner import WaypointPlanner


# Индексы в 9D observation HuskyGoalEnv:
#   0, 1    — x, y (позиция робота)
#   2, 3, 4 — vx, vy, wz
#   5, 6    — cos_yaw, sin_yaw
#   7, 8    — goal_dx, goal_dy (вектор к цели в мировой СК) ← подменяем
IDX_ROBOT_XY = slice(0, 2)
IDX_GOAL_VEC = slice(7, 9)


class HuskyGoalPlannedEnv(gym.Env):
    """Husky едет к waypoint'ам планировщика, внешне считая, что это цель."""

    metadata = HuskyGoalEnv.metadata

    def __init__(
        self,
        render_mode: str | None = None,
        n_waypoints: int = 3,
        switch_radius: float = 0.8,
        inner_env_cls: Type[HuskyGoalEnv] = HuskyGoalEnv,
    ) -> None:
        super().__init__()

        if n_waypoints < 1:
            raise ValueError(f"n_waypoints must be >= 1, got {n_waypoints}")

        # inner_env_cls должен удовлетворять контракту, описанному в docstring модуля.
        # Любой подкласс HuskyGoalEnv подходит автоматически (например, HuskyObstacleEnv).
        self._inner = inner_env_cls(render_mode=render_mode)
        self._n_waypoints = int(n_waypoints)
        self._switch_radius = float(switch_radius)

        # Прокидываем action/observation spaces 1:1 от inner env
        self.action_space = self._inner.action_space
        self.observation_space = self._inner.observation_space
        self.render_mode = self._inner.render_mode

        self._planner: WaypointPlanner | None = None

    # =========================================================================
    #   Gym API
    # =========================================================================

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self._inner.reset(seed=seed, options=options)

        start_xy = np.asarray(obs[IDX_ROBOT_XY], dtype=np.float32)
        goal_xy = np.asarray(info["goal"], dtype=np.float32)

        if self._planner is None:
            self._planner = WaypointPlanner(
                start_xy, goal_xy,
                n_waypoints=self._n_waypoints,
                switch_radius=self._switch_radius,
            )
        else:
            self._planner.reset(start_xy, goal_xy)

        obs = self._inject_waypoint_into_obs(obs)
        info = self._augment_info(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(action)
        obs = self._inject_waypoint_into_obs(obs)
        info = self._augment_info(info)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._inner.close()

    def render(self):
        return self._inner.render() if hasattr(self._inner, "render") else None

    # =========================================================================
    #   Internals
    # =========================================================================

    def _inject_waypoint_into_obs(self, obs: np.ndarray) -> np.ndarray:
        """Заменить goal_dx, goal_dy на вектор к текущему waypoint'у."""
        assert self._planner is not None, "reset() ещё не вызывался"
        robot_xy = obs[IDX_ROBOT_XY]
        wp = self._planner.current_waypoint(robot_xy)

        obs = obs.copy()
        obs[7] = float(wp[0] - robot_xy[0])
        obs[8] = float(wp[1] - robot_xy[1])
        return obs

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Добавить в info данные планировщика для логирования/визуализации."""
        if self._planner is None:
            return info
        info = dict(info)  # не мутируем чужое
        info["waypoint"] = self._planner.peek_current()
        info["waypoint_idx"] = self._planner.current_idx
        info["waypoints_all"] = self._planner.waypoints
        info["is_final_waypoint"] = self._planner.is_final()
        return info
