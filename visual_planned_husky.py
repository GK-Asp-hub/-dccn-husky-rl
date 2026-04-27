"""
GUI-визуализация TD3 + LQR + Waypoint Planner на HuskyGoalPlannedEnv.

Что рисуется:
- Красный маркер цилиндра — финальная цель (от базового env).
- Зелёный цилиндр         — АКТИВНАЯ подцель планировщика.
- Серые полупрозрачные    — ещё не активированные подцели.
- Синие полупрозрачные    — пройденные подцели (для наглядности истории).

Цвета маркеров обновляются по мере переключения планировщика,
чтобы глазами было видно, когда он «перепрыгивает» на следующую подцель.

Что смотрим:
1. Переключаются ли подцели — по смене зелёного на синий.
2. Не дёргается ли робот в момент переключения.
3. Доходит ли эпизод до финальной цели.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# torch 2.11 Windows workaround (как в других eval-скриптах)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import numpy as np
import pybullet as p
from stable_baselines3 import TD3

from envs.husky_goal_env import CONTROL_HZ, GOAL_RADIUS
from envs.husky_goal_planned_env import HuskyGoalPlannedEnv
from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions


# --- Константы отрисовки ---
WP_RADIUS = 0.25          # визуальный радиус подцели
WP_HEIGHT = 0.02
COLOR_ACTIVE = [0.2, 0.9, 0.2, 0.8]      # зелёный — текущая
COLOR_PENDING = [0.6, 0.6, 0.6, 0.35]    # серый полупрозрачный — будущая
COLOR_VISITED = [0.2, 0.4, 0.9, 0.35]    # синий полупрозрачный — пройденная


def spawn_waypoint_markers(waypoints: np.ndarray) -> list[int]:
    """Спавнит визуальные цилиндры для каждой подцели. Возвращает список их body_id."""
    body_ids = []
    for (wx, wy) in waypoints:
        visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=WP_RADIUS,
            length=WP_HEIGHT,
            rgbaColor=COLOR_PENDING,
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=[float(wx), float(wy), 0.01],
        )
        body_ids.append(body_id)
    return body_ids


def recolor_waypoints(body_ids: list[int], current_idx: int) -> None:
    """Обновляет цвета маркеров в соответствии с текущей активной подцелью."""
    for i, body_id in enumerate(body_ids):
        if i < current_idx:
            color = COLOR_VISITED
        elif i == current_idx:
            color = COLOR_ACTIVE
        else:
            color = COLOR_PENDING
        p.changeVisualShape(body_id, -1, rgbaColor=color)


def run_episode(
    env: HuskyGoalPlannedEnv,
    model: TD3,
    lqr: LQRDiffDriveLateral | None,
    alpha: float,
    seed: int,
    realtime: bool,
    slowdown: float = 1.0,
) -> dict:
    obs, info = env.reset(seed=seed)
    waypoints = info["waypoints_all"]
    start_dist = float(np.hypot(info["goal"][0], info["goal"][1]))

    # Рисуем подцели в только что пересозданной сцене
    wp_body_ids = spawn_waypoint_markers(waypoints)
    recolor_waypoints(wp_body_ids, current_idx=info["waypoint_idx"])
    last_drawn_idx = info["waypoint_idx"]

    ep_reward = 0.0
    steps = 0
    actions = []

    while True:
        # TD3 action на обогащённом obs (goal_vec указывает на активный waypoint)
        td3_action, _ = model.predict(obs, deterministic=True)

        if lqr is not None and alpha > 0:
            # LQR тоже смотрит на тот же obs — значит корректирует на АКТИВНЫЙ waypoint,
            # не на финальную цель. Это правильное поведение: LQR держит робота на линии
            # к ближайшей подцели.
            robot_x, robot_y = float(obs[0]), float(obs[1])
            cos_yaw, sin_yaw = float(obs[5]), float(obs[6])
            robot_yaw = float(np.arctan2(sin_yaw, cos_yaw))
            wp_x = robot_x + float(obs[7])
            wp_y = robot_y + float(obs[8])

            lqr_corr = lqr.compute_correction(robot_x, robot_y, robot_yaw, wp_x, wp_y)
            final_action = combine_actions(td3_action, lqr_corr, alpha=alpha)
        else:
            final_action = td3_action

        actions.append(final_action.copy())
        obs, reward, terminated, truncated, info = env.step(final_action)
        ep_reward += reward
        steps += 1

        # Если планировщик переключился — меняем цвета
        if info["waypoint_idx"] != last_drawn_idx:
            recolor_waypoints(wp_body_ids, current_idx=info["waypoint_idx"])
            last_drawn_idx = info["waypoint_idx"]

        if realtime:
            time.sleep(slowdown / CONTROL_HZ)

        if terminated or truncated:
            break

    # Пост-эпизод: перекрасим всё в «пройдено», чтобы глаз не думал, что последний wp был активным.
    # Делаем это аккуратно — только если мы действительно достигли финала.
    if info.get("reason") == "goal_reached":
        for body_id in wp_body_ids:
            p.changeVisualShape(body_id, -1, rgbaColor=COLOR_VISITED)

    actions = np.array(actions)
    smoothness = (
        float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())
        if len(actions) > 1 else 0.0
    )

    return {
        "steps": steps,
        "reward": ep_reward,
        "reason": info.get("reason", "ongoing"),
        "final_distance": info.get("distance", -1.0),
        "start_distance": start_dist,
        "goal": info["goal"],
        "action_smoothness": smoothness,
        "max_wp_idx": int(info["waypoint_idx"]),
        "n_waypoints": int(waypoints.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/husky_td3_v1_cont_best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=200,
                        help="тот же что в baseline/LQR-eval, для консистентных траекторий")
    parser.add_argument("--n-waypoints", type=int, default=3)
    parser.add_argument("--switch-radius", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="LQR residual weight. 0 = чистый TD3+planner без LQR.")
    parser.add_argument("--no-lqr", action="store_true", help="отключить LQR полностью")
    parser.add_argument("--fast", action="store_true", help="без realtime-задержки")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="множитель realtime-задержки. 1 = обычная скорость, "
                             "3 = в 3 раза медленнее. Игнорируется с --fast.")
    parser.add_argument("--headless", action="store_true",
                        help="без GUI (для прогонов на удалённой машине)")
    args = parser.parse_args()

    assert Path(args.model).exists(), f"Модель не найдена: {args.model}"

    env = HuskyGoalPlannedEnv(
        render_mode=None if args.headless else "human",
        n_waypoints=args.n_waypoints,
        switch_radius=args.switch_radius,
    )
    model = TD3.load(args.model, env=env)
    lqr = None if args.no_lqr else LQRDiffDriveLateral()

    print(f"Model:         {args.model}")
    print(f"Planner:       n_waypoints={args.n_waypoints}, switch_radius={args.switch_radius}")
    print(f"LQR alpha:     {0.0 if args.no_lqr else args.alpha}")
    print(f"Episodes:      {args.episodes}, seeds {args.seed_base}..{args.seed_base + args.episodes - 1}")
    print(f"Goal radius:   {GOAL_RADIUS} m (цель считается достигнутой если ближе)\n")

    realtime = (not args.fast) and (not args.headless)
    results = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        res = run_episode(
            env, model, lqr, alpha=args.alpha,
            seed=seed, realtime=realtime, slowdown=args.slowdown,
        )
        results.append(res)
        print(
            f"Ep {i+1}: seed={seed}  "
            f"start_dist={res['start_distance']:.2f}m  "
            f"steps={res['steps']:3d}  "
            f"reward={res['reward']:+7.2f}  "
            f"final_dist={res['final_distance']:.2f}m  "
            f"smooth={res['action_smoothness']:.4f}  "
            f"max_wp={res['max_wp_idx']}/{res['n_waypoints'] - 1}  "
            f"{res['reason']}"
        )

    env.close()

    print("\n=== Итог ===")
    success = sum(1 for r in results if r["reason"] == "goal_reached")
    rewards = np.array([r["reward"] for r in results])
    steps = np.array([r["steps"] for r in results])
    smooth = np.array([r["action_smoothness"] for r in results])
    print(f"Success: {success}/{len(results)}")
    print(f"Reward : {rewards.mean():+.2f} ± {rewards.std():.2f}")
    print(f"Steps  : {steps.mean():.1f} ± {steps.std():.1f}")
    print(f"Smooth : {smooth.mean():.4f} ± {smooth.std():.4f}")


if __name__ == "__main__":
    main()
