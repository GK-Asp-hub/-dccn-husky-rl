"""
HuskyObstacleEnv — расширение HuskyGoalEnv со статичными препятствиями
и лидар-observation'ом. Stage 2a спринта.

Что добавляется к базовому env:
- 3-5 случайных цилиндрических препятствий на арене (не на старте и не на цели)
- 16-лучевой лидар, расстояния добавляются в observation (9D → 25D)
- Плотный штраф за близость к препятствиям
- Terminal penalty + terminated=True при коллизии

Что остаётся без изменений:
- Action space (2D differential drive)
- Reward по расстоянию до цели и бонус за достижение
- Reset arena, sample goal, physics parameters
- Layout первых 9 элементов obs (важно — значит LQR и планировщик работают как раньше)

Контекст: 50_Journal/2026-04-22_planner_results.md (Stage 2 motivation).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pybullet as p
from gymnasium import spaces

from envs import husky_goal_env as _base
from envs.husky_goal_env import (
    ARENA_SIZE,
    GOAL_RADIUS,
    GOAL_SPAWN_MAX,
    GOAL_SPAWN_MIN,
    HuskyGoalEnv,
)


# --- Параметры препятствий и лидара ---
N_OBSTACLES_MIN = 3
N_OBSTACLES_MAX = 5
OBSTACLE_RADIUS_MIN = 0.3
OBSTACLE_RADIUS_MAX = 0.5
OBSTACLE_HEIGHT = 0.5

MIN_DIST_FROM_START = 1.2      # препятствие не ближе этого к (0, 0)
MIN_DIST_FROM_GOAL = 0.8       # препятствие не ближе к цели (чтобы цель была достижима)
MIN_DIST_BETWEEN_OBSTACLES = 0.6
MAX_SPAWN_ATTEMPTS = 50        # попыток найти валидную позицию до сдачи

N_LIDAR_RAYS = 16
LIDAR_MAX_RANGE = 5.0          # метры, дальше этого луч возвращает MAX_RANGE
LIDAR_HEIGHT = 0.3             # высота, на которой «светят» лучи (над плоскостью)

# --- Параметры награды, связанные с препятствиями ---
REWARD_PROXIMITY_COEF = 0.5    # плотный штраф за близость
REWARD_PROXIMITY_SAFE_DIST = 0.5  # зона безопасности (м), штраф только внутри неё
REWARD_COLLISION_PENALTY = 30.0

# Override базового step penalty для obstacle env (Stage 2a, v3+).
# Мотивация: фикс бага лидара (коммит 5e9c526) убрал фантомный штраф −0.36/шаг,
# из-за чего v2 не сошлась. Гипотеза: этот штраф работал как implicit reward
# shaping через step pressure. В v3 воспроизводим магнитуду явным step_penalty.
# См. 50_Journal/2026-04-22_v3_steppen_decision.md.
REWARD_STEP_PENALTY = 0.36


class HuskyObstacleEnv(HuskyGoalEnv):
    """Husky едет к цели через случайные статичные препятствия. Observation с лидаром."""

    def __init__(self, render_mode: str | None = None):
        super().__init__(render_mode=render_mode)

        # --- Новое observation space: 9D base + N лучей ---
        # Для первых 9 элементов bounds те же, что в HuskyGoalEnv.
        # Для лидара — диапазон [0, LIDAR_MAX_RANGE].
        base_low = self.observation_space.low
        base_high = self.observation_space.high
        lidar_low = np.zeros(N_LIDAR_RAYS, dtype=np.float32)
        lidar_high = np.full(N_LIDAR_RAYS, LIDAR_MAX_RANGE, dtype=np.float32)
        new_low = np.concatenate([base_low, lidar_low])
        new_high = np.concatenate([base_high, lidar_high])
        self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)

        # --- PyBullet-state для препятствий ---
        self._obstacle_body_ids: list[int] = []
        self._obstacle_positions: list[tuple[float, float, float]] = []  # (x, y, radius)

    # =========================================================================
    #   Lifecycle (переопределено)
    # =========================================================================

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        # Добавляем в info то же, что отдаётся в step, чтобы API был симметричным
        info["n_obstacles"] = len(self._obstacle_body_ids)
        lidar = obs[-N_LIDAR_RAYS:]
        info["min_lidar"] = float(lidar.min()) if len(lidar) > 0 else LIDAR_MAX_RANGE
        return obs, info

    def _load_world(self):
        """Расширение базового _load_world: после создания Husky'я спавним препятствия."""
        super()._load_world()
        self._spawn_obstacles()

    def _spawn_obstacles(self) -> None:
        """Случайно расставляет цилиндрические препятствия."""
        rng = self._np_random
        n = int(rng.integers(N_OBSTACLES_MIN, N_OBSTACLES_MAX + 1))

        self._obstacle_body_ids = []
        self._obstacle_positions = []

        for _ in range(n):
            placed = False
            for _attempt in range(MAX_SPAWN_ATTEMPTS):
                # Позицию ищем в пределах арены с отступом
                margin = ARENA_SIZE * 0.45
                x = float(rng.uniform(-margin, margin))
                y = float(rng.uniform(-margin, margin))
                r = float(rng.uniform(OBSTACLE_RADIUS_MIN, OBSTACLE_RADIUS_MAX))

                # Проверки
                if math.hypot(x, y) < MIN_DIST_FROM_START + r:
                    continue
                if math.hypot(x - self._goal_pos[0], y - self._goal_pos[1]) < MIN_DIST_FROM_GOAL + r:
                    continue
                too_close = False
                for (ox, oy, or_) in self._obstacle_positions:
                    if math.hypot(x - ox, y - oy) < r + or_ + MIN_DIST_BETWEEN_OBSTACLES:
                        too_close = True
                        break
                if too_close:
                    continue

                # Валидная позиция — спавним
                col = p.createCollisionShape(
                    shapeType=p.GEOM_CYLINDER, radius=r, height=OBSTACLE_HEIGHT,
                )
                vis = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER, radius=r, length=OBSTACLE_HEIGHT,
                    rgbaColor=[0.5, 0.35, 0.25, 1.0],
                )
                body_id = p.createMultiBody(
                    baseMass=0,  # статичное
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, OBSTACLE_HEIGHT / 2],
                )
                self._obstacle_body_ids.append(body_id)
                self._obstacle_positions.append((x, y, r))
                placed = True
                break
            # Если за MAX_SPAWN_ATTEMPTS не нашли — просто пропускаем это препятствие.
            # Результат: эпизод с меньшим числом препятствий, ок для нас.

    # =========================================================================
    #   Observation (переопределено)
    # =========================================================================

    def _get_observation(self) -> np.ndarray:
        base_obs = super()._get_observation()
        lidar = self._raycast_lidar()
        return np.concatenate([base_obs, lidar]).astype(np.float32)

    def _raycast_lidar(self) -> np.ndarray:
        """16 равномерно распределённых лучей вокруг робота в плоскости XY.

        Важно: фильтруем self-hits. Диагональные лучи (на углах 45°, 135°,
        225°, 315°) попадают в колёса Husky'я на расстоянии ~0.32 м, что
        раньше давало системный штраф ~0.36/шаг в reward. Теперь такие hits
        игнорируются и луч возвращает LIDAR_MAX_RANGE как «свободное
        направление».

        Fix сделан 2026-04-22 после обнаружения бага (см. debug_lidar_self.py
        и журнал 2026-04-22_lidar_bug_fix).
        """
        pos, orn = p.getBasePositionAndOrientation(self._husky_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)

        ray_from = []
        ray_to = []
        ox, oy = pos[0], pos[1]
        z = LIDAR_HEIGHT
        for i in range(N_LIDAR_RAYS):
            angle = yaw + 2.0 * math.pi * i / N_LIDAR_RAYS
            ray_from.append([ox, oy, z])
            ray_to.append([
                ox + LIDAR_MAX_RANGE * math.cos(angle),
                oy + LIDAR_MAX_RANGE * math.sin(angle),
                z,
            ])

        results = p.rayTestBatch(ray_from, ray_to)
        # results[i] = (hit_object_uid, hit_link, hit_fraction, hit_pos, hit_normal)
        # hit_fraction = 1.0 если не попал ни во что
        # Если hit_object_uid == self._husky_id (включая все его линки) — это
        # попадание в собственный корпус робота, игнорируем.
        distances = np.empty(N_LIDAR_RAYS, dtype=np.float32)
        for i, r in enumerate(results):
            hit_object_uid = r[0]
            hit_fraction = r[2]
            if hit_object_uid == self._husky_id:
                # Self-hit — считаем направление свободным
                distances[i] = LIDAR_MAX_RANGE
            else:
                distances[i] = hit_fraction * LIDAR_MAX_RANGE
        return distances

    # =========================================================================
    #   Reward (переопределено)
    # =========================================================================

    def _check_collision(self) -> bool:
        """True, если Husky в контакте с любым препятствием."""
        for obs_id in self._obstacle_body_ids:
            contacts = p.getContactPoints(bodyA=self._husky_id, bodyB=obs_id)
            if len(contacts) > 0:
                return True
        return False

    def _compute_reward_and_done(self, obs: np.ndarray):
        # Базовая логика — используем родительскую, потом модифицируем reward/termination
        reward, terminated, truncated, info = super()._compute_reward_and_done(obs)

        # Override step penalty: родитель всегда применяет _base.REWARD_STEP_PENALTY,
        # откатываем его и применяем наш (усиленный) для obstacle env.
        # См. мотивацию у определения REWARD_STEP_PENALTY выше.
        reward += _base.REWARD_STEP_PENALTY  # откат базового
        reward -= REWARD_STEP_PENALTY         # применение нашего

        # Плотный штраф за близость (по данным лидара, последние N_LIDAR_RAYS элементов obs)
        lidar = obs[-N_LIDAR_RAYS:]
        proximity_violation = np.maximum(0.0, REWARD_PROXIMITY_SAFE_DIST - lidar)
        reward -= REWARD_PROXIMITY_COEF * float(proximity_violation.sum())

        # Жёсткая коллизия → terminate + большой штраф
        # Проверяем только если ещё не завершились другой причиной
        if not terminated and not truncated:
            if self._check_collision():
                reward -= REWARD_COLLISION_PENALTY
                terminated = True
                info["reason"] = "collision"

        info["n_obstacles"] = len(self._obstacle_body_ids)
        info["min_lidar"] = float(lidar.min()) if len(lidar) > 0 else LIDAR_MAX_RANGE

        return reward, terminated, truncated, info
