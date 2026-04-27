"""Smoke test для HuskyGoalPlannedEnv. Headless, со случайными action'ами.

Проверяем, что:
- env создаётся
- shape obs совпадает с базовым (9D)
- planner инициализирован: есть waypoints в info, current_idx=0 на старте
- переключения happen: на длинном случайном роллауте индекс хоть раз подрастёт
  (не гарантировано если action'ы неудачные, но чаще всего происходит)
- inner env корректно закрывается
"""
from __future__ import annotations

import numpy as np

from envs.husky_goal_planned_env import HuskyGoalPlannedEnv


def main():
    env = HuskyGoalPlannedEnv(render_mode=None, n_waypoints=3, switch_radius=0.8)
    try:
        obs, info = env.reset(seed=42)

        # --- Sanity по shape / info
        assert obs.shape == (9,), f"obs shape {obs.shape}"
        assert obs.dtype == np.float32, f"obs dtype {obs.dtype}"
        assert "goal" in info, "info должен содержать goal (от базового env)"
        assert "waypoint" in info, "info должен содержать waypoint (от wrapper'а)"
        assert "waypoint_idx" in info
        assert "waypoints_all" in info
        assert info["waypoint_idx"] == 0, f"старт должен быть с idx=0, но {info['waypoint_idx']}"
        assert info["waypoints_all"].shape == (3, 2)
        assert not info["is_final_waypoint"]

        # goal_vec в obs должен указывать на ПЕРВЫЙ waypoint, а не на финальную цель
        robot_xy = obs[0:2]
        goal_vec = obs[7:9]
        first_wp = info["waypoints_all"][0]
        expected = first_wp - robot_xy
        assert np.allclose(goal_vec, expected, atol=1e-4), (
            f"goal_vec {goal_vec} должен быть = (first_wp - robot) {expected}"
        )

        final_goal = info["goal"]
        print(f"  start robot xy = {robot_xy}")
        print(f"  final goal     = {final_goal}")
        print(f"  waypoints      = {info['waypoints_all'].tolist()}")
        print(f"  obs[goal_vec]  = {goal_vec}  (к первому wp)")

        # --- Длинный роллаут случайных action'ов
        rng = np.random.default_rng(0)
        max_idx_seen = 0
        steps_until_done = 0
        for _ in range(300):
            action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            max_idx_seen = max(max_idx_seen, info["waypoint_idx"])
            steps_until_done += 1
            if terminated or truncated:
                break

        print(f"  steps run      = {steps_until_done}")
        print(f"  max wp idx     = {max_idx_seen}  (of {info['waypoints_all'].shape[0] - 1})")
        print(f"  terminated/truncated = {terminated}/{truncated}")
        print(f"  reason         = {info.get('reason', 'n/a')}")

        # Базовая проверка здравомыслия: или дошли до цели, или ещё ехали,
        # но env не свалился в исключение — это главный признак здоровья.
        print("\nsmoke OK")

    finally:
        env.close()


if __name__ == "__main__":
    main()
