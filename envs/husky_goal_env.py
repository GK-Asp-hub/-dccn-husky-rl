"""
HuskyGoalEnv — gymnasium-environment для задачи «доехать до цели» на Husky в PyBullet.

Design notes (см. 50_Journal/2026-04-20_husky_sanity.md):
- Action: 2D differential drive (левая пара колёс, правая пара колёс), continuous [-1, 1]
- Observation: 9D вектор (positions, velocities, orientation, goal vector in world frame)
- Reward: плотный -alpha*distance + bonus за достижение цели - step_penalty
- Termination: goal reached / timeout / upside-down
- Арена: квадрат ARENA_SIZE x ARENA_SIZE (метры), цель спавнится на расстоянии 2-5 м от старта
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


# --- Константы конфигурации ---
ARENA_SIZE = 10.0                  # сторона арены, метры
GOAL_SPAWN_MIN = 2.0               # минимальное расстояние от старта до цели
GOAL_SPAWN_MAX = 5.0               # максимальное расстояние от старта до цели
GOAL_RADIUS = 0.4                  # радиус, внутри которого цель считается достигнутой

MAX_WHEEL_VELOCITY = 10.0          # рад/с, соответствует action=+1
MAX_WHEEL_FORCE = 20.0             # Н·м, максимальный момент на колесе

SIMULATION_HZ = 240                # шаг физики, стандарт для PyBullet
CONTROL_HZ = 20                    # как часто policy выдаёт action
STEPS_PER_ACTION = SIMULATION_HZ // CONTROL_HZ   # 12 физ-шагов на один policy-шаг

MAX_EPISODE_STEPS = 500            # ~25 секунд реального времени при 20 Гц policy

# Rewards
REWARD_DISTANCE_COEF = 1.0         # alpha: чем ближе к цели, тем меньше штраф
REWARD_GOAL_BONUS = 100.0          # приз за достижение цели
REWARD_STEP_PENALTY = 0.01         # маленький штраф за каждый шаг
REWARD_FALL_PENALTY = 50.0         # штраф за переворот

# Колёсные joints Husky (проверено на sanity check)
WHEEL_JOINT_NAMES = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)


class HuskyGoalEnv(gym.Env):
    """Husky едет к случайной целевой точке на плоской арене."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        self.render_mode = render_mode

        # --- Action space ---
        # 2D differential drive: [left_wheels_velocity, right_wheels_velocity], каждый в [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # --- Observation space (9D) ---
        # [x, y, vx, vy, wz, cos_yaw, sin_yaw, goal_dx, goal_dy]
        # где goal_dx/dy - вектор от робота к цели в мировой системе координат
        obs_low = np.array(
            [-ARENA_SIZE, -ARENA_SIZE, -5, -5, -10, -1, -1, -2 * ARENA_SIZE, -2 * ARENA_SIZE],
            dtype=np.float32,
        )
        obs_high = -obs_low
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- PyBullet state ---
        self._physics_client = None
        self._husky_id = None
        self._plane_id = None
        self._goal_marker_id = None
        self._wheel_indices: list[int] = []

        # --- Episode state ---
        self._goal_pos = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_distance = 0.0  # для shaping (на будущее)

        # --- RNG (инициализируется в reset) ---
        self._np_random: np.random.Generator | None = None

    # =========================================================================
    #   Lifecycle
    # =========================================================================

    def _connect(self):
        """Ленивый коннект к PyBullet при первом reset."""
        if self._physics_client is not None:
            return
        mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self._physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _load_world(self):
        """Полный reload сцены: пол, робот, цель."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / SIMULATION_HZ)

        self._plane_id = p.loadURDF("plane.urdf")

        start_pos = [0.0, 0.0, 0.3]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self._husky_id = p.loadURDF("husky/husky.urdf", start_pos, start_orn)

        # Находим колёса
        self._wheel_indices = []
        for idx in range(p.getNumJoints(self._husky_id)):
            info = p.getJointInfo(self._husky_id, idx)
            if info[1].decode("utf-8") in WHEEL_JOINT_NAMES:
                self._wheel_indices.append(idx)
        # Порядок важен: индексы 2,3,4,5 = front_left, front_right, rear_left, rear_right
        # Проверим (на случай, если URDF когда-нибудь поменяют)
        assert len(self._wheel_indices) == 4, f"Ожидали 4 колеса, нашли {len(self._wheel_indices)}"

        # Маркер цели (визуальная сфера, без физики)
        if self.render_mode == "human":
            self._spawn_goal_marker()

    def _spawn_goal_marker(self):
        """Рисуем цель как красную полусферу на полу (только в GUI-режиме)."""
        visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=GOAL_RADIUS,
            length=0.02,
            rgbaColor=[1, 0.2, 0.2, 0.7],
        )
        self._goal_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=[self._goal_pos[0], self._goal_pos[1], 0.01],
        )

    def _sample_goal(self) -> np.ndarray:
        """Случайная цель на расстоянии [GOAL_SPAWN_MIN, GOAL_SPAWN_MAX] от старта (0,0)."""
        rng = self._np_random
        distance = rng.uniform(GOAL_SPAWN_MIN, GOAL_SPAWN_MAX)
        angle = rng.uniform(-math.pi, math.pi)
        goal = np.array(
            [distance * math.cos(angle), distance * math.sin(angle)],
            dtype=np.float32,
        )
        return goal

    # =========================================================================
    #   Gym API
    # =========================================================================

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)

        self._connect()
        self._goal_pos = self._sample_goal()
        self._load_world()

        if self.render_mode == "human":
            # Начальная камера (будет обновляться в step() следом за роботом)
            self._update_follow_camera(force_pos=[0, 0, 0])

        self._step_count = 0
        obs = self._get_observation()
        self._prev_distance = self._distance_to_goal(obs)
        info = {"goal": self._goal_pos.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        left_vel = float(action[0]) * MAX_WHEEL_VELOCITY
        right_vel = float(action[1]) * MAX_WHEEL_VELOCITY

        # Индексы колёс: [front_left, front_right, rear_left, rear_right]
        wheel_velocities = [left_vel, right_vel, left_vel, right_vel]

        for joint_idx, vel in zip(self._wheel_indices, wheel_velocities):
            p.setJointMotorControl2(
                bodyUniqueId=self._husky_id,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=MAX_WHEEL_FORCE,
            )

        # Симулируем STEPS_PER_ACTION физ-шагов = 1 policy шаг
        for _ in range(STEPS_PER_ACTION):
            p.stepSimulation()

        self._step_count += 1
        obs = self._get_observation()
        reward, terminated, truncated, info = self._compute_reward_and_done(obs)

        # Камера следует за роботом в GUI
        if self.render_mode == "human":
            self._update_follow_camera()

        return obs, reward, terminated, truncated, info

    def _update_follow_camera(self, force_pos=None):
        """Камера сверху-сзади всегда смотрит на робота."""
        if force_pos is not None:
            target = force_pos
        else:
            pos, _ = p.getBasePositionAndOrientation(self._husky_id)
            target = [pos[0], pos[1], pos[2]]
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45,
            cameraPitch=-40,
            cameraTargetPosition=target,
        )

    def close(self):
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass
            self._physics_client = None

    # =========================================================================
    #   Internals
    # =========================================================================

    def _get_observation(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(self._husky_id)
        lin_vel, ang_vel = p.getBaseVelocity(self._husky_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        goal_dx = self._goal_pos[0] - pos[0]
        goal_dy = self._goal_pos[1] - pos[1]

        return np.array(
            [
                pos[0], pos[1],
                lin_vel[0], lin_vel[1],
                ang_vel[2],
                math.cos(yaw), math.sin(yaw),
                goal_dx, goal_dy,
            ],
            dtype=np.float32,
        )

    def _distance_to_goal(self, obs: np.ndarray) -> float:
        return float(math.hypot(obs[7], obs[8]))  # goal_dx, goal_dy

    def _is_upside_down(self) -> bool:
        _, orn = p.getBasePositionAndOrientation(self._husky_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        return abs(roll) > math.radians(60) or abs(pitch) > math.radians(60)

    def _out_of_arena(self, obs: np.ndarray) -> bool:
        return abs(obs[0]) > ARENA_SIZE or abs(obs[1]) > ARENA_SIZE

    def _compute_reward_and_done(self, obs: np.ndarray):
        distance = self._distance_to_goal(obs)
        reward = -REWARD_DISTANCE_COEF * distance - REWARD_STEP_PENALTY

        terminated = False
        truncated = False
        info = {"distance": distance, "goal": self._goal_pos.copy()}

        if distance < GOAL_RADIUS:
            reward += REWARD_GOAL_BONUS
            terminated = True
            info["reason"] = "goal_reached"
        elif self._is_upside_down():
            reward -= REWARD_FALL_PENALTY
            terminated = True
            info["reason"] = "fallen"
        elif self._out_of_arena(obs):
            terminated = True
            info["reason"] = "out_of_arena"
        elif self._step_count >= MAX_EPISODE_STEPS:
            truncated = True
            info["reason"] = "timeout"

        self._prev_distance = distance
        return reward, terminated, truncated, info
