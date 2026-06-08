"""
Эксперимент B (глобальные поверхности): вся арена — одна поверхность
(NORMAL / ICE / SAND). Сравниваем control-loop'ы на каждой (табл. 2 статьи).

Для каждой пары (поверхность, layout) гоняем все 5 режимов управления и,
помимо обычных метрик успех/столкновение, собираем:
  - max_speed: пиковая линейная скорость за эпизод
  - path_ratio: длина пройденного пути / прямое расстояние старт->цель
  - smoothness: средняя ||a_t - a_{t-1}|| (гладкость управления)
Эти метрики количественно показывают проскальзывание и пере-/недоруливание.

Layout'ы:
  empty         : без препятствия, только цель в (4, 0)
  with_obstacle : та же цель, одно препятствие в (2, 0), r=0.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from stable_baselines3 import TD3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from envs.husky_surface_env import HuskySurfaceEnv, SURFACE_COLORS, SURFACE_PRESETS
from envs.husky_obstacle_env import N_LIDAR_RAYS
from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions
from controllers.obstacle_avoidance import ObstacleAvoidanceController
from controllers.map_aware_avoidance import MapAwareAvoider


LAYOUTS = {
    "empty":         {"goal": (4.0, 0.0), "obstacles": []},
    "with_obstacle": {"goal": (4.0, 0.0), "obstacles": [(2.0, 0.0, 0.5)]},
}


def run_episode(
    env, model, seed,
    use_lqr=False, lqr=None, alpha=0.0,
    use_lidar_avoid=False, lidar_avoider=None,
    use_map_avoid=False, map_avoider=None,
    max_steps=500,
):
    """Прогнать эпизод на заданной поверхности и собрать метрики (вкл. скорость и path_ratio).

    Логика выбора уровней идентична Эксп. A: активный верхний (Map/Lidar) замещает нижний+средний.
    """
    obs, info = env.reset(seed=seed)
    if lidar_avoider is not None: lidar_avoider.reset()
    if map_avoider is not None: map_avoider.reset()

    traj_x, traj_y = [], []
    actions = []
    speeds = []  # |linear velocity| each step
    avoidance_steps = 0
    ep_reward = 0.0
    steps = 0

    while True:
        rx, ry = float(obs[0]), float(obs[1])
        traj_x.append(rx); traj_y.append(ry)

        # obs[3], obs[4] — линейные vx, vy в мировой СК (см. HuskyGoalEnv); пишем |скорость| на шаге
        vx, vy = float(obs[3]), float(obs[4])
        speeds.append(math.hypot(vx, vy))

        cy_yaw, sy_yaw = float(obs[5]), float(obs[6])
        yaw = math.atan2(sy_yaw, cy_yaw)
        lidar = obs[-N_LIDAR_RAYS:]

        # Индикаторное переключение уровней (как в Эксп. A): активный верхний замещает нижний+средний.
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
            # Базовое действие нижнего уровня (TD3).
            td3_action, _ = model.predict(obs, deterministic=True)
            if use_lqr and lqr is not None and alpha > 0:
                # Средний уровень: цель в мировой СК (obs[7:9]) -> курсовая LQR-коррекция,
                gx = rx + float(obs[7])
                gy = ry + float(obs[8])
                # и остаточное сложение: a = a_TD3 + alpha*u_LQR.
                lqr_corr = lqr.compute_correction(rx, ry, yaw, gx, gy)
                final_action = combine_actions(td3_action, lqr_corr, alpha=alpha)
            else:
                final_action = td3_action
        else:
            avoidance_steps += 1

        actions.append(np.asarray(final_action).copy())

        obs, reward, terminated, truncated, info = env.step(final_action)
        ep_reward += float(reward)
        steps += 1
        if terminated or truncated or steps >= max_steps:
            break

    actions = np.asarray(actions)
    # Гладкость управления: средняя норма приращения действия (меньше = плавнее руль).
    smoothness = (
        float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())
        if len(actions) > 1 else 0.0
    )

    # Длина пути и её отношение к прямому расстоянию (path_ratio > 1 = крюки/проскальзывание).
    tx = np.asarray(traj_x); ty = np.asarray(traj_y)
    if len(tx) > 1:
        seg = np.hypot(np.diff(tx), np.diff(ty))
        path_len = float(seg.sum())
    else:
        path_len = 0.0
    goal_dx = env._goal_param[0] - 0.0
    goal_dy = env._goal_param[1] - 0.0
    direct = float(math.hypot(goal_dx, goal_dy))
    path_ratio = path_len / direct if direct > 1e-6 else float("inf")

    return {
        "seed": seed,
        "steps": steps,
        "reward": ep_reward,
        "reason": info.get("reason", "ongoing"),
        "smoothness": smoothness,
        "max_speed": float(max(speeds)) if speeds else 0.0,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "path_length": path_len,
        "path_ratio": path_ratio,
        "trajectory_x": traj_x,
        "trajectory_y": traj_y,
        "success": info.get("reason") == "goal_reached",
        "collision": info.get("reason") == "collision",
        "avoidance_steps": avoidance_steps,
    }


def plot_one(surface, layout_name, goal, obstacles, results_by_label, out_path):
    fig, ax = plt.subplots(figsize=(11, 7))

    # Whole-arena tinted background
    bg_col = SURFACE_COLORS.get(surface, [0.6, 0.6, 0.6, 0.4])
    bg = MplCircle((0, 0), 8.0, color=bg_col, alpha=0.4, zorder=0)
    ax.add_patch(bg)

    for (ox, oy, orad) in obstacles:
        ax.add_patch(MplCircle((ox, oy), orad, color="#7c5a3a", alpha=0.9, zorder=2))

    ax.plot(*goal, marker="*", markersize=24, color="#dc2626",
            label="Goal", linestyle="None", zorder=3)
    ax.plot(0, 0, marker="o", markersize=10, color="#16a34a",
            label="Start", linestyle="None", zorder=3)

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
                    color=color, alpha=0.8, linewidth=1.8,
                    linestyle=ls, label=tag, zorder=4)

    margin = 1.5
    ax.set_xlim(-margin, max(goal[0], 4.0) + margin)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    params = SURFACE_PRESETS[surface]
    ax.set_title(
        f"Surface: {surface}  ({layout_name})    "
        f"mu_lat={params['lateralFriction']}, mu_roll={params['rollingFriction']}"
    )
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, _ in uniq], [l for _, l in uniq],
              loc="best", fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  [plot] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/husky_obstacle_td3_v3_steppen036_best/best_model.zip")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="figures/expB_global")
    parser.add_argument("--surfaces", type=str, default="NORMAL,ICE,SAND")
    parser.add_argument("--layouts", type=str, default="empty,with_obstacle")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    model = TD3.load(args.model)
    lqr = LQRDiffDriveLateral()
    lidar_avoider = ObstacleAvoidanceController()
    map_avoider = MapAwareAvoider()

    surface_list = [s.strip() for s in args.surfaces.split(",")]
    layout_list  = [l.strip() for l in args.layouts.split(",")]
    seeds        = [args.seed_base + i for i in range(args.seeds)]

    configs = [
        ("TD3",              dict()),
        ("TD3+LQR",          dict(use_lqr=True, lqr=lqr, alpha=args.alpha)),
        ("TD3+LidarAvoid",   dict(use_lidar_avoid=True, lidar_avoider=lidar_avoider)),
        ("TD3+MapAvoid",     dict(use_map_avoid=True, map_avoider=map_avoider)),
        ("TD3+MapAvoid+LQR", dict(use_map_avoid=True, map_avoider=map_avoider,
                                  use_lqr=True, lqr=lqr, alpha=args.alpha)),
    ]

    summary = {}  # (surface, layout) -> {label: stats}

    for surface in surface_list:
        for layout_name in layout_list:
            if layout_name not in LAYOUTS:
                continue
            spec = LAYOUTS[layout_name]
            goal = spec["goal"]; obstacles = spec["obstacles"]
            print(f"\n{'='*72}")
            print(f"  {surface}  /  {layout_name}  goal={goal}  n_obs={len(obstacles)}")
            print('=' * 72)

            results_by_label = {}
            for label, kw in configs:
                env = HuskySurfaceEnv(
                    render_mode=None,
                    goal=goal,
                    obstacles=obstacles,
                    global_surface=surface,
                )
                runs = []
                for s in seeds:
                    r = run_episode(env, model, seed=s, **kw)
                    runs.append(r)
                env.close()
                results_by_label[label] = runs
                succ = sum(r["success"] for r in runs)
                coll = sum(r["collision"] for r in runs)
                print(f"  {label:<18} success={succ}/{len(runs)}  coll={coll}/{len(runs)}  "
                      f"steps={np.mean([r['steps'] for r in runs]):.0f}  "
                      f"max_v={np.mean([r['max_speed'] for r in runs]):.2f}  "
                      f"path/direct={np.mean([r['path_ratio'] for r in runs]):.2f}  "
                      f"smooth={np.mean([r['smoothness'] for r in runs]):.3f}")

            summary[f"{surface}__{layout_name}"] = {
                label: {
                    "success": sum(r["success"] for r in runs),
                    "collision": sum(r["collision"] for r in runs),
                    "n": len(runs),
                    "avg_steps": float(np.mean([r["steps"] for r in runs])),
                    "avg_reward": float(np.mean([r["reward"] for r in runs])),
                    "avg_smoothness": float(np.mean([r["smoothness"] for r in runs])),
                    "avg_max_speed": float(np.mean([r["max_speed"] for r in runs])),
                    "avg_path_ratio": float(np.mean([r["path_ratio"] for r in runs])),
                }
                for label, runs in results_by_label.items()
            }

            tag = f"{surface}_{layout_name}"
            plot_one(surface, layout_name, goal, obstacles,
                      results_by_label, out_dir / f"{tag}_trajectories.png")

            json_path = out_dir / f"{tag}_results.json"
            json_path.write_text(json.dumps({
                "surface": surface,
                "layout": layout_name,
                "goal": list(goal),
                "obstacles": obstacles,
                "presets": SURFACE_PRESETS[surface],
                "summary": summary[f"{surface}__{layout_name}"],
                "runs": {label: [{k: v for k, v in r.items()
                                  if k not in ("trajectory_x", "trajectory_y")}
                                 for r in runs]
                         for label, runs in results_by_label.items()},
            }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Cross-surface summary table
    print("\n" + "=" * 100)
    print("  SUMMARY: success / max_speed (mean over seeds)")
    print("=" * 100)
    config_labels = [c[0] for c in configs]
    header = f"{'Surface/Layout':<28}" + "".join(f"{lbl:>15}" for lbl in config_labels)
    print(header); print("-" * len(header))
    for surface in surface_list:
        for layout in layout_list:
            key = f"{surface}__{layout}"
            if key not in summary: continue
            row = f"{surface}/{layout:<14}"[:28].ljust(28)
            for lbl in config_labels:
                s = summary[key][lbl]
                cell = f"{s['success']}/{s['n']} v={s['avg_max_speed']:.1f}"
                row += f"{cell:>15}"
            print(row)
    print("=" * 100)

    agg_path = out_dir / "aggregate_results.json"
    agg_path.write_text(json.dumps({
        "config": vars(args),
        "summary": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] aggregate -> {agg_path}")


if __name__ == "__main__":
    main()
