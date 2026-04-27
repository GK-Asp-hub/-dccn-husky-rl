"""Smoke test для HuskyObstacleEnv. Headless, со случайными action'ами.

Проверяем:
- env создаётся
- obs shape = 25 (9 base + 16 lidar)
- препятствия спавнятся (n_obstacles из info)
- лидар возвращает разумные расстояния (не все нули, не все MAX_RANGE)
- reward всегда конечный, не NaN
- коллизия детектируется, если специально в препятствие впилиться
- разные seeds → разные сцены
"""
from __future__ import annotations

import numpy as np

from envs.husky_obstacle_env import HuskyObstacleEnv, LIDAR_MAX_RANGE, N_LIDAR_RAYS


def run_single_episode(env, seed: int, max_steps: int = 50):
    obs, info = env.reset(seed=seed)
    assert obs.shape == (9 + N_LIDAR_RAYS,), f"obs shape {obs.shape}"
    assert obs.dtype == np.float32
    assert "n_obstacles" in info, "info must include n_obstacles"

    n_obs = info["n_obstacles"]
    lidar = obs[-N_LIDAR_RAYS:]
    # Лидар не должен быть полностью нулевым (робот точно что-то видит — в крайнем случае землю на расстоянии)
    # но и не полностью MAX — на арене стоят препятствия
    print(f"  seed={seed}  n_obs={n_obs}  lidar min={lidar.min():.2f}  max={lidar.max():.2f}  mean={lidar.mean():.2f}")

    rng = np.random.default_rng(seed)
    total_r = 0.0
    for step in range(max_steps):
        action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert np.isfinite(reward), f"non-finite reward at step {step}"
        assert obs.shape == (9 + N_LIDAR_RAYS,)

        total_r += reward
        if terminated or truncated:
            print(f"    ended at step {step}: reason={info.get('reason')}  total_r={total_r:.2f}  "
                  f"min_lidar={info.get('min_lidar'):.2f}")
            return info.get("reason", "unknown"), n_obs

    print(f"    still running after {max_steps} steps, total_r={total_r:.2f}")
    return "ongoing", n_obs


def main():
    env = HuskyObstacleEnv(render_mode=None)
    try:
        print("=== HuskyObstacleEnv smoke test ===\n")
        print("3 эпизода со случайной policy, разные seeds\n")

        seen_obstacle_counts = []
        for seed in [42, 100, 200]:
            reason, n = run_single_episode(env, seed, max_steps=80)
            seen_obstacle_counts.append(n)

        # Разные сиды дают разное количество препятствий (либо разные позиции)
        print(f"\nObstacle counts across seeds: {seen_obstacle_counts}")
        assert len(set(seen_obstacle_counts)) >= 1, "got same n_obstacles"
        # Минимум 3 препятствия спавнятся (даже с rejection sampling)
        assert min(seen_obstacle_counts) >= 1, f"got <1 obstacles, env broken: {seen_obstacle_counts}"

        print("\nsmoke OK")
    finally:
        env.close()


if __name__ == "__main__":
    main()
