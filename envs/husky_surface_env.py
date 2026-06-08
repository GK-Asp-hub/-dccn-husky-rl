"""
HuskySurfaceEnv -- HuskyMultiObstacleEnv плюс зоны трения поверхности.

Добавляет на пол круглые зоны, где меняется боковое трение контакта колёс с грунтом.
Три преднастроенных типа поверхности (фактические значения — в SURFACE_PRESETS ниже):

  NORMAL : боковое трение 0.8  (обычный грунт / асфальт)
  ICE    : боковое трение 0.15, без сопротивления качению  (скользко, сносит в поворотах)
  SAND   : боковое трение 0.9 + сопротивление качению 0.05  (вязко, медленно)

Базовый пол остаётся на дефолте PyBullet. Зоны рисуются плоскими цветными кругами,
их трение применяется к колёсам, попавшим внутрь.

Замечание по реализации: PyBullet не выставляет «трение участка пола» напрямую.
Аппроксимируем — на каждом шаге управления меняем динамику КОЛЁС в зависимости от
того, в какой зоне робот (или базовый пол, если вне зон). Для дифпривода на
однородном грунте это корректно; случай «одно колесо на льду, другое на асфальте»
игнорируется (требовал бы поконтактного учёта). Для демонстрации — простое правило
«решает центр робота».
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pybullet as p

from envs.husky_multi_obstacle_env import HuskyMultiObstacleEnv


# Коэффициенты трения по умолчанию (боковое + сопротивление качению).
# ВАЖНО: контактное трение в PyBullet — произведение (или min) коэффициентов двух тел,
# поэтому заданное здесь значение напрямую определяет сцепление колесо-грунт.
SURFACE_PRESETS = {
    "NORMAL": {"lateralFriction": 0.8, "rollingFriction": 0.01},
    # ICE: продольного сцепления хватает катиться, бокового почти нет -> робота сносит в
    # поворотах. PyBullet не разделяет боковое/продольное трение шины, поэтому держим
    # умеренное mu (0.15) и за счёт нулевого качения + слабого бокового удержания
    # получаем проскальзывание в резких манёврах.
    "ICE":    {"lateralFriction": 0.15, "rollingFriction": 0.0},
    # SAND: продольное сцепление как у нормали, но сильное сопротивление качению ->
    # робот едет, но вяло стартует и медленно поворачивает.
    "SAND":   {"lateralFriction": 0.9, "rollingFriction": 0.05},
}

# Цвета зон для визуализации
SURFACE_COLORS = {
    "NORMAL": [0.6, 0.6, 0.6, 0.5],
    "ICE":    [0.7, 0.85, 1.0, 0.7],
    "SAND":   [0.85, 0.75, 0.45, 0.7],
}


class HuskySurfaceEnv(HuskyMultiObstacleEnv):
    """Среда с препятствиями, расширенная зонами трения поверхности.

    Параметры
    ---------
    surface_zones : список (x, y, radius, kind)
        Каждая зона — круг на полу в (x, y) заданного радиуса; kind из {"NORMAL","ICE","SAND"}.

    Остальные параметры: см. HuskyMultiObstacleEnv.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        goal: tuple[float, float] = (4.0, 0.0),
        obstacles: Sequence[tuple[float, float, float]] = (),
        surface_zones: Sequence[tuple[float, float, float, str]] = (),
        global_surface: str | None = None,
    ):
        """Если задан `global_surface` (напр. "ICE"), трение применяется к колёсам весь
        эпизод независимо от позиции, и весь пол перекрашивается; `surface_zones` тогда
        игнорируется. Именно этот режим используется в Эксп. B (глобальная поверхность).
        """
        self._global_surface = (
            None if global_surface is None else str(global_surface)
        )
        self._surface_zones = [
            (float(x), float(y), float(r), str(kind))
            for (x, y, r, kind) in surface_zones
        ]
        self._zone_visual_ids: list[int] = []
        self._ground_overlay_id: int | None = None
        self._current_surface = self._global_surface or "NORMAL"
        super().__init__(render_mode=render_mode, goal=goal, obstacles=obstacles)

    # ---------- Spawning visual zone markers ----------

    def _load_world(self):
        super()._load_world()
        self._zone_visual_ids = []
        self._ground_overlay_id = None

        if self._global_surface is not None:
            # Single global surface: cover the whole arena with a big
            # tinted disc and apply the friction once.
            color = SURFACE_COLORS.get(self._global_surface, [0.5, 0.5, 0.5, 0.6])
            vis = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=15.0,             # large enough to cover the whole arena
                length=0.005,
                rgbaColor=color,
            )
            self._ground_overlay_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=vis,
                basePosition=[0.0, 0.0, 0.003],
            )
            self._current_surface = self._global_surface
        else:
            # Local zones mode (legacy): spawn per-zone visual markers
            for (x, y, r, kind) in self._surface_zones:
                color = SURFACE_COLORS.get(kind, [0.5, 0.5, 0.5, 0.5])
                vis = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=r,
                    length=0.005,
                    rgbaColor=color,
                )
                body = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, 0.003],
                )
                self._zone_visual_ids.append(body)
            self._current_surface = "NORMAL"

        # Apply current surface friction to wheels
        self._apply_surface_to_wheels(self._current_surface)

    # ---------- Surface application ----------

    def _surface_at(self, robot_x: float, robot_y: float) -> str:
        """Return surface kind at the given world position."""
        for (zx, zy, zr, kind) in self._surface_zones:
            if (robot_x - zx) ** 2 + (robot_y - zy) ** 2 <= zr * zr:
                return kind
        return "NORMAL"

    def _apply_surface_to_wheels(self, kind: str) -> None:
        """Apply friction params to BOTH wheels and the ground plane.

        PyBullet contact friction is computed pair-wise from the two body
        coefficients. Changing only the wheels is not enough -- the plane
        keeps its default high friction and the contact stays grippy. We
        write the same value to the plane base (link -1) so both sides
        of the contact agree on the desired friction.
        """
        params = SURFACE_PRESETS.get(kind, SURFACE_PRESETS["NORMAL"])
        for joint_idx in self._wheel_indices:
            p.changeDynamics(
                self._husky_id,
                joint_idx,
                lateralFriction=params["lateralFriction"],
                rollingFriction=params["rollingFriction"],
            )
        # Plane has body id self._plane_id, link -1 means the base link.
        if hasattr(self, "_plane_id") and self._plane_id is not None:
            p.changeDynamics(
                self._plane_id,
                -1,
                lateralFriction=params["lateralFriction"],
                rollingFriction=params["rollingFriction"],
            )

    # ---------- Step override: detect surface and re-apply ----------

    def step(self, action):
        # Global mode: friction was set once in _load_world, no per-step
        # work needed.
        if self._global_surface is not None:
            return super().step(action)

        # Local zones mode: re-detect surface every step based on robot
        # centre and re-apply friction if it changed.
        pos, _ = p.getBasePositionAndOrientation(self._husky_id)
        kind = self._surface_at(float(pos[0]), float(pos[1]))
        if kind != self._current_surface:
            self._apply_surface_to_wheels(kind)
            self._current_surface = kind
        return super().step(action)

    # ---------- Public access for video / debug ----------

    @property
    def current_surface(self) -> str:
        return self._current_surface


# ---------------- Predefined scenarios with surfaces ----------------

SURFACE_SCENARIOS = {
    "ice_strip": {
        "goal": (4.0, 0.0),
        "obstacles": [],
        "surface_zones": [(2.0, 0.0, 1.0, "ICE")],
    },
    "ice_strip_with_obstacle": {
        "goal": (5.0, 0.0),
        "obstacles": [(3.5, 0.0, 0.5)],
        "surface_zones": [(1.7, 0.0, 0.9, "ICE")],
    },
    "sand_strip": {
        "goal": (4.0, 0.0),
        "obstacles": [],
        "surface_zones": [(2.0, 0.0, 1.0, "SAND")],
    },
    "sand_strip_with_obstacle": {
        "goal": (5.0, 0.0),
        "obstacles": [(3.5, 0.0, 0.5)],
        "surface_zones": [(1.7, 0.0, 0.9, "SAND")],
    },
    "ice_and_sand": {
        # Ice patch first, then sand patch -- mixed challenge
        "goal": (5.5, 0.0),
        "obstacles": [],
        "surface_zones": [
            (1.6, 0.0, 0.7, "ICE"),
            (3.6, 0.0, 0.7, "SAND"),
        ],
    },
    "obstacle_then_ice": {
        "goal": (5.5, 0.0),
        "obstacles": [(2.0, 0.0, 0.5)],
        "surface_zones": [(4.0, 0.0, 1.0, "ICE")],
    },
}
