"""
Эксперимент A (финальный прогон): детерминированное препятствие на пути, сравнение 5 режимов.

Режимы (все используют ОДНУ и ту же обученную модель TD3, без переобучения):
  1. TD3                  -- только нижний реактивный уровень (базовая политика)
  2. TD3 + LQR            -- остаточная схема с классическим регулятором (нижний + средний)
  3. TD3 + LidarAvoid     -- эвристический обход по сырому лидару, без карты (сенсорный верхний)
  4. TD3 + MapAvoid       -- обход по известной карте (модельный верхний)
  5. TD3 + MapAvoid + LQR -- полная тройка (верхний доминирует, когда активен)

По каждому режиму считается: доля успеха, доля столкновений, число шагов, награда,
гладкость управления и доля эпизода в режиме обхода.

Также строит график траекторий всех режимов (рис. 2 статьи).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import math
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from envs.husky_obstacle_deterministic_env import HuskyObstacleDeterministicEnv
from envs.husky_obstacle_env import N_LIDAR_RAYS
from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions
from controllers.obstacle_avoidance import ObstacleAvoidanceController
from controllers.map_aware_avoidance import MapAwareAvoider


def run_episode(
    env, model, seed,
    use_lqr=False, lqr=None, alpha=0.0,
    use_lidar_avoid=False, lidar_avoider=None,
    use_map_avoid=False, map_avoider=None,
    max_steps=500,
):
    """Прогнать один эпизод с заданной комбинацией уровней и собрать метрики.

    Флаги use_* включают соответствующие уровни; на каждом шаге индикатор верхнего
    уровня (MapAvoid/LidarAvoid) решает, кто формирует действие.
    """
    obs, info = env.reset(seed=seed)
    if lidar_avoider is not None: lidar_avoider.reset()
    if map_avoider is not None: map_avoider.reset()

    traj_x, traj_y = [], []
    actions = []
    avoidance_steps = 0
    ep_reward = 0.0
    steps = 0
    min_lidar = float("inf")

    while True:
        traj_x.append(float(obs[0]))
        traj_y.append(float(obs[1]))

        rx, ry = float(obs[0]), float(obs[1])
        cy_yaw, sy_yaw = float(obs[5]), float(obs[6])
        yaw = math.atan2(sy_yaw, cy_yaw)
        lidar = obs[-N_LIDAR_RAYS:]

        # --- Выбор источника действия (индикаторное переключение уровней) ---
        # Если активен верхний уровень (Map/Lidar), он ЗАМЕЩАЕТ нижний+средний.
        in_avoid = False
        if use_map_avoid and map_avoider is not None:
            obstacles = env._obstacle_positions
            if map_avoider.should_take_over(rx, ry, yaw, obstacles):
                final_action = map_avoider.compute_action(rx, ry, yaw, obstacles)
                in_avoid = True
        elif use_lidar_avoid and lidar_avoider is not None:
            if lidar_avoider.should_take_over(lidar):
                final_action = lidar_avoider.compute_action(lidar)
                in_avoid = True

        if not in_avoid:
            # Верхний уровень неактивен -> базовое действие даёт нижний уровень (TD3).
            td3_action, _ = model.predict(obs, deterministic=True)
            if use_lqr and lqr is not None and alpha > 0:
                # Средний уровень: восстановить цель в мировой СК (obs[7:9] — вектор к цели),
                gx = rx + float(obs[7])
                gy = ry + float(obs[8])
                # посчитать курсовую LQR-коррекцию и сложить остаточно: a = a_TD3 + alpha*u_LQR.
                lqr_corr = lqr.compute_correction(rx, ry, yaw, gx, gy)
                final_action = combine_actions(td3_action, lqr_corr, alpha=alpha)
            else:
                final_action = td3_action
        else:
            # Шаг прошёл под управлением верхнего уровня (для метрики avoidance_fraction).
            avoidance_steps += 1

        actions.append(np.asarray(final_action).copy())

        obs, reward, terminated, truncated, info = env.step(final_action)
        ep_reward += float(reward)
        steps += 1
        if "min_lidar" in info:
            min_lidar = min(min_lidar, float(info["min_lidar"]))

        if terminated or truncated or steps >= max_steps:
            break

    actions = np.asarray(actions)
    # Гладкость управления: средняя норма приращения действия между шагами
    # (меньше = плавнее руль; именно эту метрику улучшает средний уровень LQR).
    smoothness = (
        float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())
        if len(actions) > 1 else 0.0
    )

    return {
        "seed": seed,
        "steps": steps,
        "reward": ep_reward,
        "reason": info.get("reason", "ongoing"),
        "final_distance": float(info.get("distance", -1.0)),
        "min_lidar": min_lidar if min_lidar != float("inf") else None,
        "smoothness": smoothness,
        "trajectory_x": traj_x,
        "trajectory_y": traj_y,
        "success": info.get("reason") == "goal_reached",
        "collision": info.get("reason") == "collision",
        "avoidance_steps": avoidance_steps,
        "avoidance_fraction": avoidance_steps / max(steps, 1),
    }


def plot_trajectories(results_by_label, obstacle_x, obstacle_y, obstacle_r,
                       goal_x, goal_y, out_path):
    fig, ax = plt.subplots(figsize=(11, 7))
    obs_circle = plt.Circle((obstacle_x, obstacle_y), obstacle_r,
                             color="#7c5a3a", alpha=0.85, label=f"Obstacle (r={obstacle_r})")
    ax.add_patch(obs_circle)
    ax.plot(goal_x, goal_y, marker="*", markersize=24,
            color="#dc2626", label="Goal", linestyle="None")
    ax.plot(0, 0, marker="o", markersize=10,
            color="#16a34a", label="Start", linestyle="None")

    palette = {
        "TD3":              "#2563eb",
        "TD3+LQR":          "#ea580c",
        "TD3+LidarAvoid":   "#16a34a",
        "TD3+MapAvoid":     "#a855f7",
        "TD3+MapAvoid+LQR": "#0891b2",
    }
    for label, runs in results_by_label.items():
        color = palette.get(label, "#000000")
        for i, r in enumerate(runs):
            tag = label if i == 0 else None
            ls = "-" if r["success"] else "--"
            ax.plot(r["trajectory_x"], r["trajectory_y"],
                    color=color, alpha=0.8, linewidth=2.0,
                    linestyle=ls, label=tag)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1, max(goal_x + 2, 6))
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Experiment A: deterministic obstacle on the line to goal\n"
                 "(solid = success, dashed = collision)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[ok] Saved trajectory plot: {out_path}")


def summarize(label, runs):
    succ = sum(r["success"] for r in runs)
    coll = sum(r["collision"] for r in runs)
    n = len(runs)
    return {
        "label": label,
        "n": n,
        "success": f"{succ}/{n}",
        "collision": f"{coll}/{n}",
        "avg_steps": float(np.mean([r["steps"] for r in runs])),
        "avg_reward": float(np.mean([r["reward"] for r in runs])),
        "avg_smoothness": float(np.mean([r["smoothness"] for r in runs])),
        "avg_avoid_frac": float(np.mean([r["avoidance_fraction"] for r in runs])),
    }


def print_summary(s):
    print(f"\n{'-'*72}")
    print(f"  {s['label']}")
    print(f"{'-'*72}")
    print(f"    success      : {s['success']}")
    print(f"    collision    : {s['collision']}")
    print(f"    avg steps    : {s['avg_steps']:.1f}")
    print(f"    avg reward   : {s['avg_reward']:.2f}")
    print(f"    smoothness   : {s['avg_smoothness']:.4f}")
    if s['avg_avoid_frac'] > 0:
        print(f"    avoid frac   : {s['avg_avoid_frac']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/husky_obstacle_td3_v3_steppen036_best/best_model.zip")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--goal-distance", type=float, default=4.0)
    parser.add_argument("--obstacle-x", type=float, default=2.0)
    parser.add_argument("--obstacle-y-offset", type=float, default=0.0)
    parser.add_argument("--obstacle-radius", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument("--tag", type=str, default="expA_final")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  EXPERIMENT A FINAL  --  obstacle ({args.obstacle_x},{args.obstacle_y_offset}) r={args.obstacle_radius}")
    print("=" * 72)

    print(f"\n[load] {args.model}")
    model = TD3.load(args.model)
    lqr = LQRDiffDriveLateral()
    lidar_avoider = ObstacleAvoidanceController()
    map_avoider = MapAwareAvoider()

    seeds = [args.seed_base + i for i in range(args.seeds)]

    def make_env():
        # Детерминированная среда: препятствие в фиксированной точке -> воспроизводимый исход.
        return HuskyObstacleDeterministicEnv(
            render_mode=None,
            goal_distance=args.goal_distance,
            obstacle_x=args.obstacle_x,
            obstacle_y_offset=args.obstacle_y_offset,
            obstacle_radius=args.obstacle_radius,
        )

    # Пять сравниваемых режимов (табл. 1): от чистого нижнего уровня до полной тройки.
    configs = [
        ("TD3",              dict()),
        ("TD3+LQR",          dict(use_lqr=True, lqr=lqr, alpha=args.alpha)),
        ("TD3+LidarAvoid",   dict(use_lidar_avoid=True, lidar_avoider=lidar_avoider)),
        ("TD3+MapAvoid",     dict(use_map_avoid=True, map_avoider=map_avoider)),
        ("TD3+MapAvoid+LQR", dict(use_map_avoid=True, map_avoider=map_avoider,
                                  use_lqr=True, lqr=lqr, alpha=args.alpha)),
    ]

    results_by_label = {}
    for label, kw in configs:
        print(f"\n[run]  {label}")
        env = make_env()
        runs = []
        for s in seeds:
            r = run_episode(env, model, seed=s, **kw)
            runs.append(r)
            print(f"    seed={s}:  {r['reason']:<14}  steps={r['steps']:>3}  "
                  f"reward={r['reward']:>8.2f}  avoid_frac={r['avoidance_fraction']*100:>5.1f}%")
        env.close()
        results_by_label[label] = runs

    summaries = [summarize(label, runs) for label, runs in results_by_label.items()]
    print("\n" + "=" * 72 + "\n  SUMMARY")
    for s in summaries:
        print_summary(s)
    print("=" * 72)

    plot_path = out_dir / f"{args.tag}_trajectories.png"
    plot_trajectories(
        results_by_label,
        obstacle_x=args.obstacle_x,
        obstacle_y=args.obstacle_y_offset,
        obstacle_r=args.obstacle_radius,
        goal_x=args.goal_distance,
        goal_y=0.0,
        out_path=str(plot_path),
    )

    json_path = out_dir / f"{args.tag}_results.json"
    payload = {
        "config": vars(args),
        "summaries": summaries,
        "runs": {label: [{k: v for k, v in r.items()
                          if k not in ("trajectory_x", "trajectory_y")}
                         for r in runs]
                 for label, runs in results_by_label.items()},
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
