"""
WaypointPlanner — простой goal-conditioned планировщик для мобильного робота.

Разбивает прямую от start к goal на N равных сегментов и по очереди выдаёт
промежуточные точки (waypoints) как активные подцели. Переключение на
следующий waypoint происходит, когда робот оказывается в радиусе switch_radius
от текущего.

Используется как обёртка над уже обученной TD3-политикой: TD3 умеет ехать
к произвольной точке (goal_vector есть в observation), а planner просто
подменяет эту точку на ближайший waypoint в observation. Переобучать TD3
не требуется.

Контекст: см. 50_Journal/2026-04-21_planner_decision.md.
"""

from __future__ import annotations

import numpy as np


class WaypointPlanner:
    """Линейный интерполирующий планировщик.

    Делит отрезок start→goal на `n_waypoints` равных сегментов, выдаёт подцели
    в порядке start→...→goal. Последняя подцель всегда совпадает с финальной
    целью.

    Parameters
    ----------
    start_xy : array-like, shape (2,)
        Начальная позиция робота в мировой системе координат.
    goal_xy : array-like, shape (2,)
        Финальная цель в мировой системе координат.
    n_waypoints : int
        Количество подцелей. n=1 → одна подцель = финальная цель (planner-no-op).
        n=2 → середина + финал. n=3 → три равных сегмента, и т.д.
    switch_radius : float
        Радиус переключения (метры). Когда робот входит в круг этого радиуса
        вокруг текущей подцели, planner переключается на следующую. Для
        последней подцели переключения не происходит.
    """

    def __init__(
        self,
        start_xy,
        goal_xy,
        n_waypoints: int,
        switch_radius: float,
    ) -> None:
        if n_waypoints < 1:
            raise ValueError(f"n_waypoints must be >= 1, got {n_waypoints}")
        if switch_radius <= 0:
            raise ValueError(f"switch_radius must be > 0, got {switch_radius}")

        self.n_waypoints = int(n_waypoints)
        self.switch_radius = float(switch_radius)

        self._waypoints: np.ndarray = np.zeros((self.n_waypoints, 2), dtype=np.float32)
        self._current_idx: int = 0

        self.reset(start_xy, goal_xy)

    def reset(self, start_xy, goal_xy) -> None:
        """Пересобрать список waypoint'ов под новый эпизод."""
        start = np.asarray(start_xy, dtype=np.float32).reshape(2)
        goal = np.asarray(goal_xy, dtype=np.float32).reshape(2)

        # Линейная интерполяция: генерируем n точек,
        # (i+1)-я точка = start + (i+1)/n * (goal - start), i = 0..n-1.
        # Таким образом первая подцель уже смещена от start, а последняя = goal.
        ts = np.linspace(1.0 / self.n_waypoints, 1.0, self.n_waypoints, dtype=np.float32)
        self._waypoints = start[None, :] + ts[:, None] * (goal - start)[None, :]
        self._current_idx = 0

    def current_waypoint(self, robot_xy) -> np.ndarray:
        """Вернуть активную подцель, при необходимости переключить вперёд.

        Если робот находится в пределах `switch_radius` от текущей подцели
        и эта подцель не последняя — переключаемся на следующую. Для последней
        подцели переключения не происходит (иначе «потеряем» финальную цель).
        """
        robot = np.asarray(robot_xy, dtype=np.float32).reshape(2)
        current = self._waypoints[self._current_idx]
        distance = float(np.linalg.norm(robot - current))

        if distance < self.switch_radius and self._current_idx < self.n_waypoints - 1:
            self._current_idx += 1
            current = self._waypoints[self._current_idx]

        return current.copy()

    def peek_current(self) -> np.ndarray:
        """Вернуть активную подцель БЕЗ переключений и без side effects.

        Используется для логирования и визуализации, когда обновлять индекс
        нельзя (`current_waypoint` уже была вызвана на этом шаге).
        """
        return self._waypoints[self._current_idx].copy()

    def is_final(self) -> bool:
        """True, если активная подцель — последняя (финальная)."""
        return self._current_idx == self.n_waypoints - 1

    @property
    def current_idx(self) -> int:
        """Индекс активной подцели (0-based). Полезно для логирования и визуализации."""
        return self._current_idx

    @property
    def waypoints(self) -> np.ndarray:
        """Массив всех подцелей, shape (n_waypoints, 2). Read-only-контракт."""
        return self._waypoints.copy()
