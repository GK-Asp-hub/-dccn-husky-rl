"""Smoke-test: HuskyGoalPlannedEnv оборачивает HuskyObstacleEnv."""
from __future__ import annotations

import numpy as np

from envs.husky_goal_planned_env import HuskyGoalPlannedEnv
from envs.husky_obstacle_env import HuskyObstacleEnv, N_LIDAR_RAYS


def main():
    env = HuskyGoalPlannedEnv(
        render_mode=None,
        n_waypoints=3,
        switch_radius=0.8,
        inner_env_cls=HuskyObstacleEnv,  # новый параметр
    )
    try:
        obs, info = env.reset(seed=42)
        assert obs.shape == (9 + N_LIDAR_RAYS,), f"obs shape {obs.shape}"
        assert "waypoint" in info, "planner info missing"
        assert "n_obstacles" in info, "obstacle info missing (inner env info pass-through broken?)"
        assert info["waypoints_all"].shape == (3, 2)

        print(f"  obs shape: {obs.shape}")
        print(f"  first 9 (base):    {obs[:9].round(3)}")
        print(f"  last 16 (lidar):   min={obs[-16:].min():.2f}  max={obs[-16:].max():.2f}")
        print(f"  info keys: {sorted(info.keys())}")
        print(f"  n_obstacles: {info['n_obstacles']}")
        print(f"  waypoints:   {info['waypoints_all'].tolist()}")

        # Быстрый roll-out, проверим что step тоже пропускает info от inner
        rng = np.random.default_rng(0)
        for _ in range(30):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=(2,)).astype(np.float32))
            assert obs.shape == (9 + N_LIDAR_RAYS,)
            assert np.isfinite(r)
            if term or trunc:
                break

        print(f"\n  final: reward seen OK, obs shape stable, planner at idx={info['waypoint_idx']}")
        print("\nsmoke OK")
    finally:
        env.close()


if __name__ == "__main__":
    main()
