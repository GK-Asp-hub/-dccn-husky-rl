"""
Визуализация траекторий TD3 vs TD3+LQR vs TD3+LQR+Planner на одних сидах.
Прогон в headless-режиме, запись (x, y) робота каждый policy-step,
построение matplotlib-графика, сохранение PNG в figures/.

Третья линия (planner) опциональна: флаг --no-planner отключает её и
воспроизводит старое поведение скрипта (2 линии).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# torch 2.11 Windows workaround
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import matplotlib

matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import TD3

from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions
from envs.husky_goal_env import GOAL_RADIUS, HuskyGoalEnv
from envs.husky_goal_planned_env import HuskyGoalPlannedEnv


def record_trajectory_pure(env: HuskyGoalEnv, model: TD3, seed: int):
    obs, info = env.reset(seed=seed)
    xs = [float(obs[0])]
    ys = [float(obs[1])]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        xs.append(float(obs[0]))
        ys.append(float(obs[1]))
        if terminated or truncated:
            break
    return np.array(xs), np.array(ys), info["goal"], info.get("reason", "?")


def record_trajectory_lqr(env: HuskyGoalEnv, model: TD3, lqr: LQRDiffDriveLateral, alpha: float, seed: int):
    obs, info = env.reset(seed=seed)
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
    return np.array(xs), np.array(ys), info["goal"], info.get("reason", "?")


def record_trajectory_planner(
    env: HuskyGoalPlannedEnv,
    model: TD3,
    lqr: LQRDiffDriveLateral,
    alpha: float,
    seed: int,
):
    """Прогон TD3 + LQR + waypoint-планировщик. Дополнительно возвращает waypoints."""
    obs, info = env.reset(seed=seed)
    waypoints = np.array(info["waypoints_all"], dtype=np.float32)  # снапшот (не меняется за эпизод)
    xs = [float(obs[0])]
    ys = [float(obs[1])]
    while True:
        td3_action, _ = model.predict(obs, deterministic=True)
        rx, ry = float(obs[0]), float(obs[1])
        cy, sy = float(obs[5]), float(obs[6])
        yaw = float(np.arctan2(sy, cy))
        # В planner-env obs[7:9] уже указывает на активный waypoint,
        # так что LQR автоматически корректирует на него.
        wx = rx + float(obs[7])
        wy = ry + float(obs[8])
        corr = lqr.compute_correction(rx, ry, yaw, wx, wy)
        final_action = combine_actions(td3_action, corr, alpha=alpha)
        obs, _, terminated, truncated, info = env.step(final_action)
        xs.append(float(obs[0]))
        ys.append(float(obs[1]))
        if terminated or truncated:
            break
    return np.array(xs), np.array(ys), info["goal"], info.get("reason", "?"), waypoints


def plot_grid(
    seeds,
    trajectories_pure,
    trajectories_lqr,
    trajectories_planner,  # список кортежей или None если --no-planner
    save_path: Path,
    n_waypoints_label: int | None = None,
):
    n = len(seeds)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)

    for i, seed in enumerate(seeds):
        ax = axes[i // cols][i % cols]
        xs_p, ys_p, goal_p, reason_p = trajectories_pure[i]
        xs_l, ys_l, goal_l, reason_l = trajectories_lqr[i]
        assert np.allclose(goal_p, goal_l), "Goals must match for the same seed"

        ax.plot(xs_p, ys_p, "-", color="tab:blue", linewidth=2.0,
                label=f"TD3 [{reason_p}, {len(xs_p)-1}st]")
        ax.plot(xs_l, ys_l, "-", color="tab:orange", linewidth=2.0,
                label=f"TD3+LQR [{reason_l}, {len(xs_l)-1}st]")

        if trajectories_planner is not None:
            xs_pl, ys_pl, goal_pl, reason_pl, wps = trajectories_planner[i]
            assert np.allclose(goal_p, goal_pl), "Goals must match for the same seed"
            label_pl = (
                f"+Planner N={n_waypoints_label} [{reason_pl}, {len(xs_pl)-1}st]"
                if n_waypoints_label is not None
                else f"+Planner [{reason_pl}, {len(xs_pl)-1}st]"
            )
            ax.plot(xs_pl, ys_pl, "-", color="tab:green", linewidth=2.0, label=label_pl)
            # Waypoint'ы — зелёные крестики, финальный (=goal) уже рисуется красным ниже,
            # поэтому рисуем только промежуточные (всё кроме последнего).
            if len(wps) > 1:
                ax.plot(
                    wps[:-1, 0], wps[:-1, 1],
                    "x", color="tab:green", markersize=10, markeredgewidth=2,
                    label="waypoints", alpha=0.7,
                )

        ax.plot(0, 0, "ks", markersize=8, label="start")

        goal_circle = plt.Circle(
            (float(goal_p[0]), float(goal_p[1])),
            GOAL_RADIUS,
            color="tab:red",
            alpha=0.25,
            label=f"goal (r={GOAL_RADIUS})",
        )
        ax.add_patch(goal_circle)
        ax.plot(float(goal_p[0]), float(goal_p[1]), "r+", markersize=12)

        ax.set_title(f"seed={seed}")
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    # Отключить пустые оси (если seeds не кратно cols)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    if trajectories_planner is not None:
        title = f"Trajectories: TD3 vs TD3+LQR vs TD3+LQR+Planner(N={n_waypoints_label}), same seeds"
    else:
        title = "Trajectories: TD3 vs TD3+LQR (same seeds)"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/husky_td3_v1_cont_best/best_model.zip")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[200, 202, 205, 206, 207, 209],
                        help="смесь для статьи: быстрые (207,209), длинные/сложные "
                             "(205 reverse, 206 hard), нормальные (200, 202).")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--n-waypoints", type=int, default=3,
                        help="N waypoints для planner-прогона. Из ablation'а N=3 "
                             "и N=2 показали одинаковый success (9/10). N=3 нагляднее.")
    parser.add_argument("--switch-radius", type=float, default=0.8)
    parser.add_argument("--no-planner", action="store_true",
                        help="воспроизвести старую версию без третьей линии")
    parser.add_argument("--out", type=str, default="figures/trajectories_td3_vs_lqr_vs_planner.png")
    args = parser.parse_args()

    # --- Plain env для TD3 и TD3+LQR ---
    env_plain = HuskyGoalEnv(render_mode=None)
    model_plain = TD3.load(args.model, env=env_plain)
    lqr = LQRDiffDriveLateral()

    print(f"Model:         {args.model}")
    print(f"Seeds:         {args.seeds}")
    print(f"alpha:         {args.alpha}")
    print(f"planner:       {'off' if args.no_planner else f'N={args.n_waypoints}, sw={args.switch_radius}'}")
    print()

    traj_pure = []
    traj_lqr = []
    for seed in args.seeds:
        p = record_trajectory_pure(env_plain, model_plain, seed)
        l = record_trajectory_lqr(env_plain, model_plain, lqr, args.alpha, seed)
        print(f"seed={seed:3d}  TD3 steps={len(p[0])-1:3d} ({p[3]:13s})   "
              f"TD3+LQR steps={len(l[0])-1:3d} ({l[3]})")
        traj_pure.append(p)
        traj_lqr.append(l)
    env_plain.close()

    # --- Planner env, если включено ---
    traj_planner = None
    if not args.no_planner:
        env_planned = HuskyGoalPlannedEnv(
            render_mode=None,
            n_waypoints=args.n_waypoints,
            switch_radius=args.switch_radius,
        )
        model_planned = TD3.load(args.model, env=env_planned)
        traj_planner = []
        for seed in args.seeds:
            t = record_trajectory_planner(env_planned, model_planned, lqr, args.alpha, seed)
            print(f"seed={seed:3d}  +Planner(N={args.n_waypoints}) "
                  f"steps={len(t[0])-1:3d} ({t[3]})")
            traj_planner.append(t)
        env_planned.close()

    plot_grid(
        args.seeds, traj_pure, traj_lqr, traj_planner,
        Path(args.out),
        n_waypoints_label=None if args.no_planner else args.n_waypoints,
    )


if __name__ == "__main__":
    main()
