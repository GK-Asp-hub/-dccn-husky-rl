"""
Визуализация траекторий для Stage 2a: TD3 vs TD3+LQR vs TD3+LQR+Planner
на HuskyObstacleEnv с препятствиями + лидар.

Генерирует рис. 5.2 для статьи — аналог plot_trajectories.py (Stage 1),
но с отрисовкой препятствий как серых кругов на каждом subplot.

Показательные seeds для статьи:
- 304 — decisive case: без планировщика timeout 500, с N=3 — 78 шагов
- 300, 302 — нормальные случаи для сравнения
- 303, 306, 309 — простые случаи, все режимы справляются

Usage:
    python plot_trajectories_obstacle.py
    python plot_trajectories_obstacle.py --seeds 304 300 309 --n-waypoints 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# torch 2.11 Windows workaround
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from stable_baselines3 import TD3

from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions
from envs.husky_goal_env import GOAL_RADIUS
from envs.husky_goal_planned_env import HuskyGoalPlannedEnv
from envs.husky_obstacle_env import HuskyObstacleEnv


# =============================================================================
#   Episode runners — записывают (x, y) робота и позиции препятствий
# =============================================================================

def _snapshot_obstacles(env) -> list[tuple[float, float, float]]:
    """Достаём позиции препятствий из env. Работаем как с обычным HuskyObstacleEnv,
    так и с обёрткой HuskyGoalPlannedEnv (тогда env._inner — HuskyObstacleEnv)."""
    target = env
    if hasattr(env, "_inner"):
        target = env._inner
    return list(getattr(target, "_obstacle_positions", []))


def record_pure_td3(env: HuskyObstacleEnv, model: TD3, seed: int):
    obs, info = env.reset(seed=seed)
    obstacles = _snapshot_obstacles(env)
    xs = [float(obs[0])]
    ys = [float(obs[1])]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        xs.append(float(obs[0]))
        ys.append(float(obs[1]))
        if terminated or truncated:
            break
    return np.array(xs), np.array(ys), info["goal"], info.get("reason", "?"), obstacles


def record_td3_lqr(env: HuskyObstacleEnv, model: TD3, lqr: LQRDiffDriveLateral,
                   alpha: float, seed: int):
    obs, info = env.reset(seed=seed)
    obstacles = _snapshot_obstacles(env)
    xs = [float(obs[0])]
    ys = [float(obs[1])]
    while True:
        td3_action, _ = model.predict(obs, deterministic=True)
        rx, ry = float(obs[0]), float(obs[1])
        cy, sy = float(obs[5]), float(obs[6])
        yaw = float(np.arctan2(sy, cy))
        gx = rx + float(obs[7])
        gy = ry + float(obs[8])
        corr = lqr.compute_correction(rx, ry, yaw, gx, gy)
        final_action = combine_actions(td3_action, corr, alpha=alpha)
        obs, _, terminated, truncated, info = env.step(final_action)
        xs.append(float(obs[0]))
        ys.append(float(obs[1]))
        if terminated or truncated:
            break
    return np.array(xs), np.array(ys), info["goal"], info.get("reason", "?"), obstacles


def record_planner(env: HuskyGoalPlannedEnv, model: TD3, lqr: LQRDiffDriveLateral,
                   alpha: float, seed: int):
    obs, info = env.reset(seed=seed)
    obstacles = _snapshot_obstacles(env)
    waypoints = np.array(info["waypoints_all"], dtype=np.float32)
    xs = [float(obs[0])]
    ys = [float(obs[1])]
    while True:
        td3_action, _ = model.predict(obs, deterministic=True)
        rx, ry = float(obs[0]), float(obs[1])
        cy, sy = float(obs[5]), float(obs[6])
        yaw = float(np.arctan2(sy, cy))
        wx = rx + float(obs[7])
        wy = ry + float(obs[8])
        corr = lqr.compute_correction(rx, ry, yaw, wx, wy)
        final_action = combine_actions(td3_action, corr, alpha=alpha)
        obs, _, terminated, truncated, info = env.step(final_action)
        xs.append(float(obs[0]))
        ys.append(float(obs[1]))
        if terminated or truncated:
            break
    return np.array(xs), np.array(ys), info["goal"], info.get("reason", "?"), obstacles, waypoints


# =============================================================================
#   Plotting
# =============================================================================

def plot_grid(seeds, traj_pure, traj_lqr, traj_planner, save_path: Path,
              n_waypoints_label: int):
    n = len(seeds)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)

    for i, seed in enumerate(seeds):
        ax = axes[i // cols][i % cols]
        xs_p, ys_p, goal_p, reason_p, obstacles_p = traj_pure[i]
        xs_l, ys_l, goal_l, reason_l, _ = traj_lqr[i]
        xs_pl, ys_pl, goal_pl, reason_pl, _, wps = traj_planner[i]

        # Все три режима используют одну и ту же обученную модель и один и тот же env seed,
        # поэтому препятствия одинаковые. Берём из pure-прогона.
        for (ox, oy, r) in obstacles_p:
            circle = Circle(
                (ox, oy), r,
                facecolor="lightgray", edgecolor="dimgray",
                linewidth=1.2, alpha=0.7, zorder=1,
            )
            ax.add_patch(circle)

        # Старт
        ax.plot(0, 0, "ks", markersize=9, zorder=5)

        # Финальная цель
        goal_circle = Circle(
            (float(goal_p[0]), float(goal_p[1])),
            GOAL_RADIUS,
            facecolor="tab:red", alpha=0.22, edgecolor="tab:red",
            linewidth=1.2, zorder=2,
        )
        ax.add_patch(goal_circle)
        ax.plot(float(goal_p[0]), float(goal_p[1]), "r+", markersize=14, zorder=5)

        # Три траектории
        ax.plot(xs_p, ys_p, "-", color="tab:blue", linewidth=2.0, zorder=3,
                label=f"TD3 [{reason_p}, {len(xs_p)-1}st]")
        ax.plot(xs_l, ys_l, "-", color="tab:orange", linewidth=2.0, zorder=3,
                label=f"TD3+LQR [{reason_l}, {len(xs_l)-1}st]")
        ax.plot(xs_pl, ys_pl, "-", color="tab:green", linewidth=2.0, zorder=4,
                label=f"+Planner N={n_waypoints_label} [{reason_pl}, {len(xs_pl)-1}st]")

        # Waypoints (кроме последнего — он и так красный финал)
        if len(wps) > 1:
            ax.plot(
                wps[:-1, 0], wps[:-1, 1],
                "x", color="tab:green", markersize=11, markeredgewidth=2.2,
                label="waypoints", alpha=0.8, zorder=5,
            )

        ax.set_title(f"seed={seed}  (n_obs={len(obstacles_p)})", fontsize=11)
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        # Немного расширим границы, чтобы все препятствия вошли
        all_x = np.concatenate([xs_p, xs_l, xs_pl, [o[0] for o in obstacles_p], [float(goal_p[0]), 0]])
        all_y = np.concatenate([ys_p, ys_l, ys_pl, [o[1] for o in obstacles_p], [float(goal_p[1]), 0]])
        pad = 0.5
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(
        f"Stage 2a: TD3 vs TD3+LQR vs TD3+LQR+Planner(N={n_waypoints_label}) "
        f"with random obstacles",
        fontsize=13,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")


# =============================================================================
#   Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/husky_obstacle_td3_v3_steppen036_best/best_model.zip")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[301, 304, 303, 306, 308, 309],
                        help="Seeds из ablation Stage 2a v3. 301 — механизм 1 (waypoint в опасной зоне, коллизия), 304 — механизм 2 (planner ломает финишный манёвр, timeout).")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--n-waypoints", type=int, default=3)
    parser.add_argument("--switch-radius", type=float, default=0.8)
    parser.add_argument("--out", type=str,
                        default="figures/trajectories_obstacle_v3_N3.png")
    args = parser.parse_args()

    assert Path(args.model).exists(), f"Модель не найдена: {args.model}"

    # === Pure / LQR env ===
    env_obs = HuskyObstacleEnv(render_mode=None)
    model_plain = TD3.load(args.model, env=env_obs)
    lqr = LQRDiffDriveLateral()

    print(f"Model:         {args.model}")
    print(f"Seeds:         {args.seeds}")
    print(f"alpha:         {args.alpha}")
    print(f"N waypoints:   {args.n_waypoints}")
    print()

    traj_pure, traj_lqr = [], []
    for seed in args.seeds:
        p = record_pure_td3(env_obs, model_plain, seed)
        l = record_td3_lqr(env_obs, model_plain, lqr, args.alpha, seed)
        print(f"seed={seed:3d}  TD3 steps={len(p[0])-1:3d} ({p[3]:13s})   "
              f"TD3+LQR steps={len(l[0])-1:3d} ({l[3]})")
        traj_pure.append(p)
        traj_lqr.append(l)
    env_obs.close()

    # === Planner env ===
    env_planned = HuskyGoalPlannedEnv(
        render_mode=None,
        n_waypoints=args.n_waypoints,
        switch_radius=args.switch_radius,
        inner_env_cls=HuskyObstacleEnv,
    )
    model_planned = TD3.load(args.model, env=env_planned)

    traj_planner = []
    for seed in args.seeds:
        t = record_planner(env_planned, model_planned, lqr, args.alpha, seed)
        print(f"seed={seed:3d}  +Planner(N={args.n_waypoints}) "
              f"steps={len(t[0])-1:3d} ({t[3]})")
        traj_planner.append(t)
    env_planned.close()

    plot_grid(
        args.seeds, traj_pure, traj_lqr, traj_planner,
        Path(args.out), n_waypoints_label=args.n_waypoints,
    )


if __name__ == "__main__":
    main()
