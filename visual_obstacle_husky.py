"""
GUI-визуализация для Stage 2a: TD3 (+ опционально LQR и Planner) на
HuskyObstacleEnv. Рисует:
- Базовую арену и Husky (от PyBullet).
- Случайные препятствия (цилиндры).
- 16 лучей лидара как жёлтые линии из робота — что он «видит».
- Waypoints как цветные цилиндры (если --planner).
- Красный цилиндр финальной цели.

Что смотрим глазами:
1. Агент действительно избегает столкновений с препятствиями? (Цель Stage 2a.)
2. Лидар-лучи меняются реалистично с движением — длинные в пустом
   направлении, короткие когда нос упирается в препятствие?
3. Планировщик (если --planner) даёт преимущество в обходе? Или ломает
   стратегию как в пустой арене?

Usage:
    python visual_obstacle_husky.py --episodes 2 --slowdown 3
    python visual_obstacle_husky.py --episodes 2 --slowdown 3 --planner --n-waypoints 3
    python visual_obstacle_husky.py --headless --fast   # для тестов без GUI
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# torch 2.11 Windows workaround
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import numpy as np
import pybullet as p
from stable_baselines3 import TD3

from envs.husky_goal_env import CONTROL_HZ, GOAL_RADIUS
from envs.husky_obstacle_env import HuskyObstacleEnv, N_LIDAR_RAYS, LIDAR_MAX_RANGE, LIDAR_HEIGHT
from envs.husky_goal_planned_env import HuskyGoalPlannedEnv
from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions


# --- Цвета и параметры отрисовки ---
WP_RADIUS = 0.25
WP_HEIGHT = 0.02
COLOR_ACTIVE = [0.2, 0.9, 0.2, 0.8]      # зелёный — активный waypoint
COLOR_PENDING = [0.6, 0.6, 0.6, 0.35]    # серый — будущий
COLOR_VISITED = [0.2, 0.4, 0.9, 0.35]    # синий — пройденный

LIDAR_RAY_COLOR_HIT = [1.0, 0.8, 0.1]     # жёлтый — попал во что-то
LIDAR_RAY_COLOR_MISS = [0.4, 0.6, 1.0]    # голубой — свободное направление


def spawn_waypoint_markers(waypoints: np.ndarray) -> list[int]:
    body_ids = []
    for (wx, wy) in waypoints:
        visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=WP_RADIUS, length=WP_HEIGHT,
            rgbaColor=COLOR_PENDING,
        )
        body_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=visual,
            basePosition=[float(wx), float(wy), 0.01],
        )
        body_ids.append(body_id)
    return body_ids


def recolor_waypoints(body_ids: list[int], current_idx: int) -> None:
    for i, body_id in enumerate(body_ids):
        if i < current_idx:
            color = COLOR_VISITED
        elif i == current_idx:
            color = COLOR_ACTIVE
        else:
            color = COLOR_PENDING
        p.changeVisualShape(body_id, -1, rgbaColor=color)


def draw_lidar_rays(obs_lidar: np.ndarray, robot_pos: tuple[float, float],
                    robot_yaw: float, prev_line_ids: list[int]) -> list[int]:
    """Рисуем 16 лучей по данным лидара. Возвращаем новые line_ids,
    старые нужно удалить перед вызовом (removeUserDebugItem)."""
    # Удаляем предыдущие
    for lid in prev_line_ids:
        try:
            p.removeUserDebugItem(lid)
        except Exception:
            pass

    new_ids = []
    ox, oy = robot_pos
    z = LIDAR_HEIGHT
    for i in range(N_LIDAR_RAYS):
        angle = robot_yaw + 2.0 * np.pi * i / N_LIDAR_RAYS
        d = float(obs_lidar[i])
        # Выбираем цвет по тому, попал ли луч (d < MAX_RANGE = попал)
        is_hit = d < LIDAR_MAX_RANGE * 0.99
        color = LIDAR_RAY_COLOR_HIT if is_hit else LIDAR_RAY_COLOR_MISS
        end_x = ox + d * np.cos(angle)
        end_y = oy + d * np.sin(angle)
        lid = p.addUserDebugLine(
            [ox, oy, z], [end_x, end_y, z],
            lineColorRGB=color, lineWidth=1.5,
        )
        new_ids.append(lid)
    return new_ids


def run_episode(
    env,  # HuskyObstacleEnv или HuskyGoalPlannedEnv(inner=HuskyObstacleEnv)
    model: TD3,
    lqr: LQRDiffDriveLateral | None,
    alpha: float,
    use_planner: bool,
    seed: int,
    realtime: bool,
    slowdown: float,
    draw_lidar: bool,
    setup_camera: bool = False,
    record_path: str | None = None,
) -> dict:
    obs, info = env.reset(seed=seed)

    # После первого reset PyBullet client поднят — здесь можно настраивать
    # камеру и запускать запись видео.
    if setup_camera:
        try:
            p.resetDebugVisualizerCamera(
                cameraDistance=8.0, cameraYaw=45.0, cameraPitch=-50.0,
                cameraTargetPosition=[0.0, 0.0, 0.0],
            )
        except Exception as e:
            print(f"[warn] не удалось выставить камеру: {e}")

    record_log_id = None
    if record_path is not None:
        Path(record_path).parent.mkdir(parents=True, exist_ok=True)
        record_log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, record_path,
        )
        print(f"Recording:       {record_path} (logId={record_log_id})")

    start_dist = float(np.hypot(info["goal"][0], info["goal"][1]))

    # Если с планировщиком — рисуем waypoints
    wp_body_ids: list[int] = []
    last_drawn_idx = 0
    if use_planner:
        waypoints = info["waypoints_all"]
        wp_body_ids = spawn_waypoint_markers(waypoints)
        recolor_waypoints(wp_body_ids, current_idx=info["waypoint_idx"])
        last_drawn_idx = info["waypoint_idx"]

    # Debug-lines лидара обновляются каждый шаг
    lidar_line_ids: list[int] = []

    ep_reward = 0.0
    steps = 0
    actions = []
    min_lidar = float("inf")

    while True:
        td3_action, _ = model.predict(obs, deterministic=True)

        if lqr is not None and alpha > 0:
            robot_x, robot_y = float(obs[0]), float(obs[1])
            cos_yaw, sin_yaw = float(obs[5]), float(obs[6])
            robot_yaw = float(np.arctan2(sin_yaw, cos_yaw))
            gx = robot_x + float(obs[7])
            gy = robot_y + float(obs[8])
            lqr_corr = lqr.compute_correction(robot_x, robot_y, robot_yaw, gx, gy)
            final_action = combine_actions(td3_action, lqr_corr, alpha=alpha)
        else:
            final_action = td3_action

        actions.append(final_action.copy())

        # Отрисовываем лидар до step (по текущему obs)
        if draw_lidar:
            robot_xy = (float(obs[0]), float(obs[1]))
            cy, sy = float(obs[5]), float(obs[6])
            yaw = float(np.arctan2(sy, cy))
            lidar_line_ids = draw_lidar_rays(
                obs[-N_LIDAR_RAYS:], robot_xy, yaw, lidar_line_ids,
            )

        obs, reward, terminated, truncated, info = env.step(final_action)
        ep_reward += reward
        steps += 1
        if "min_lidar" in info:
            min_lidar = min(min_lidar, info["min_lidar"])

        # Перекрашивание waypoint при переключении
        if use_planner and info.get("waypoint_idx", last_drawn_idx) != last_drawn_idx:
            recolor_waypoints(wp_body_ids, current_idx=info["waypoint_idx"])
            last_drawn_idx = info["waypoint_idx"]

        if realtime:
            time.sleep(slowdown / CONTROL_HZ)

        if terminated or truncated:
            break

    # Зачищаем debug-линии лидара в конце эпизода
    for lid in lidar_line_ids:
        try:
            p.removeUserDebugItem(lid)
        except Exception:
            pass

    actions = np.array(actions)
    smoothness = (
        float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())
        if len(actions) > 1 else 0.0
    )

    if record_log_id is not None:
        p.stopStateLogging(record_log_id)
        print(f"Saved video:     {record_path}")

    return {
        "steps": steps,
        "reward": ep_reward,
        "reason": info.get("reason", "ongoing"),
        "final_distance": info.get("distance", -1.0),
        "start_distance": start_dist,
        "min_lidar_seen": min_lidar if min_lidar != float("inf") else LIDAR_MAX_RANGE,
        "n_obstacles": info.get("n_obstacles", 0),
        "smoothness": smoothness,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/husky_obstacle_td3_v2_best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.2, help="LQR residual weight")
    parser.add_argument("--no-lqr", action="store_true", help="отключить LQR")
    parser.add_argument("--planner", action="store_true",
                        help="использовать waypoint-планировщик поверх")
    parser.add_argument("--n-waypoints", type=int, default=3)
    parser.add_argument("--switch-radius", type=float, default=0.8)
    parser.add_argument("--no-lidar-draw", action="store_true",
                        help="не рисовать лучи лидара в GUI (меньше визуального шума)")
    parser.add_argument("--fast", action="store_true", help="без realtime-задержки")
    parser.add_argument("--slowdown", type=float, default=1.0,
                        help="множитель realtime-задержки. >1 = медленнее")
    parser.add_argument("--headless", action="store_true", help="без GUI (для тестов)")
    parser.add_argument("--record", type=str, default=None,
                        help="путь к выходному mp4 для записи окна PyBullet (требует ffmpeg в PATH)")
    args = parser.parse_args()

    assert Path(args.model).exists(), f"Модель не найдена: {args.model}"

    render_mode = None if args.headless else "human"
    if args.planner:
        env = HuskyGoalPlannedEnv(
            render_mode=render_mode,
            n_waypoints=args.n_waypoints,
            switch_radius=args.switch_radius,
            inner_env_cls=HuskyObstacleEnv,
        )
    else:
        env = HuskyObstacleEnv(render_mode=render_mode)

    model = TD3.load(args.model, env=env)
    lqr = None if args.no_lqr else LQRDiffDriveLateral()

    realtime = (not args.fast) and (not args.headless)
    draw_lidar = (not args.no_lidar_draw) and (not args.headless)

    print(f"Model:           {args.model}")
    print(f"Mode:            TD3 {'only' if args.no_lqr else '+ LQR'}"
          f"{' + Planner (N='+str(args.n_waypoints)+')' if args.planner else ''}")
    print(f"LQR alpha:       {0.0 if args.no_lqr else args.alpha}")
    print(f"GUI:             {'off (headless)' if args.headless else 'on'}")
    print(f"Lidar overlay:   {'on' if draw_lidar else 'off'}")
    print(f"Episodes:        {args.episodes}, seeds {args.seed_base}..{args.seed_base + args.episodes - 1}")
    print()

    if args.record and args.headless:
        print("[warn] --record несовместим с --headless, запись будет пропущена")

    try:
        results = []
        for i in range(args.episodes):
            seed = args.seed_base + i
            # Камеру и запись настраиваем только в первом эпизоде.
            is_first = (i == 0)
            res = run_episode(
                env, model, lqr, args.alpha,
                use_planner=args.planner, seed=seed,
                realtime=realtime, slowdown=args.slowdown, draw_lidar=draw_lidar,
                setup_camera=is_first and not args.headless,
                record_path=(args.record if is_first and args.record and not args.headless else None),
            )
            results.append(res)
            print(
                f"Ep {i+1}: seed={seed}  n_obs={res['n_obstacles']}  "
                f"start={res['start_distance']:.2f}m  "
                f"steps={res['steps']:3d}  "
                f"reward={res['reward']:+7.2f}  "
                f"final_dist={res['final_distance']:.2f}m  "
                f"min_lidar={res['min_lidar_seen']:.2f}  "
                f"{res['reason']}"
            )

        # Сводка
        print("\n=== Итог ===")
        success = sum(1 for r in results if r["reason"] == "goal_reached")
        collisions = sum(1 for r in results if r["reason"] == "collision")
        rewards = np.array([r["reward"] for r in results])
        print(f"Success:    {success}/{len(results)}")
        print(f"Collisions: {collisions}/{len(results)}")
        print(f"Reward:     {rewards.mean():+.2f} +/- {rewards.std():.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
