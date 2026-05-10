"""
Experiment A video recording -- MP4 via PyBullet GUI, same format as
videos in the public repo.

Uses PyBullet's STATE_LOGGING_VIDEO_MP4 which routes the GUI window
output through ffmpeg (which must be in PATH). One run per
configuration, all five configurations recorded in sequence.

Output: figures/expA_videos_mp4/0N_<config>.mp4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import numpy as np
import pybullet as p
from stable_baselines3 import TD3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from envs.husky_obstacle_deterministic_env import HuskyObstacleDeterministicEnv
from envs.husky_obstacle_env import N_LIDAR_RAYS
from envs.husky_goal_env import CONTROL_HZ
from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions
from controllers.obstacle_avoidance import ObstacleAvoidanceController
from controllers.map_aware_avoidance import MapAwareAvoider


def setup_camera_topdown():
    """Top-down chase camera centred between start and goal."""
    p.resetDebugVisualizerCamera(
        cameraDistance=6.5,
        cameraYaw=0,        # looking straight along +X
        cameraPitch=-65,    # mostly top-down
        cameraTargetPosition=[2.0, 0.0, 0.0],
    )


def update_follow_camera(robot_pos):
    """Follow camera that tracks the robot from above-behind."""
    p.resetDebugVisualizerCamera(
        cameraDistance=4.0,
        cameraYaw=45,
        cameraPitch=-50,
        cameraTargetPosition=[robot_pos[0], robot_pos[1], 0.0],
    )


def run_with_recording(
    env, model, seed, record_path,
    use_lqr=False, lqr=None, alpha=0.0,
    use_lidar_avoid=False, lidar_avoider=None,
    use_map_avoid=False, map_avoider=None,
    max_steps=500,
    realtime_factor=2.0,
    follow_camera=True,
):
    obs, info = env.reset(seed=seed)
    if lidar_avoider is not None: lidar_avoider.reset()
    if map_avoider is not None: map_avoider.reset()

    # Camera before recording starts so initial frame is good
    if follow_camera:
        update_follow_camera([0.0, 0.0, 0.0])
    else:
        setup_camera_topdown()

    # Start MP4 logging
    Path(record_path).parent.mkdir(parents=True, exist_ok=True)
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, record_path)
    print(f"  recording -> {record_path}  (logId={log_id})")

    # Small pause so the first frame includes the static scene clearly
    for _ in range(int(CONTROL_HZ * 0.5)):
        p.stepSimulation()
        time.sleep(realtime_factor / CONTROL_HZ)

    steps = 0
    avoidance_steps = 0
    while True:
        rx, ry = float(obs[0]), float(obs[1])
        cy_yaw, sy_yaw = float(obs[5]), float(obs[6])
        yaw = math.atan2(sy_yaw, cy_yaw)
        lidar = obs[-N_LIDAR_RAYS:]

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
            td3_action, _ = model.predict(obs, deterministic=True)
            if use_lqr and lqr is not None and alpha > 0:
                gx = rx + float(obs[7])
                gy = ry + float(obs[8])
                lqr_corr = lqr.compute_correction(rx, ry, yaw, gx, gy)
                final_action = combine_actions(td3_action, lqr_corr, alpha=alpha)
            else:
                final_action = td3_action
        else:
            avoidance_steps += 1

        obs, reward, terminated, truncated, info = env.step(final_action)
        steps += 1

        if follow_camera:
            update_follow_camera([rx, ry, 0.0])

        # Real-time playback so the recorded video is watchable
        time.sleep(realtime_factor / CONTROL_HZ)

        if terminated or truncated or steps >= max_steps:
            break

    # Linger ~1 s on the final frame so the outcome is visible
    for _ in range(int(CONTROL_HZ * 1.0)):
        p.stepSimulation()
        time.sleep(realtime_factor / CONTROL_HZ)

    p.stopStateLogging(log_id)
    print(f"  saved.  steps={steps}  avoid_frac={avoidance_steps/max(steps,1)*100:.1f}%  "
          f"reason={info.get('reason')}")

    return {
        "steps": steps,
        "avoidance_steps": avoidance_steps,
        "reason": info.get("reason", "ongoing"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/husky_obstacle_td3_v3_steppen036_best/best_model.zip")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--goal-distance", type=float, default=4.0)
    parser.add_argument("--obstacle-x", type=float, default=2.0)
    parser.add_argument("--obstacle-y-offset", type=float, default=0.0)
    parser.add_argument("--obstacle-radius", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="figures/expA_videos_mp4")
    parser.add_argument("--realtime-factor", type=float, default=1.5,
                        help=">1 means slower than real-time playback in the recording")
    parser.add_argument("--top-down", action="store_true",
                        help="static top-down camera instead of follow")
    parser.add_argument("--only", type=str, default=None,
                        help="run only a single config by name (e.g. 04_TD3_MapAvoid)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}")
    model = TD3.load(args.model)
    lqr = LQRDiffDriveLateral()
    lidar_avoider = ObstacleAvoidanceController()
    map_avoider = MapAwareAvoider()

    configs = [
        ("01_TD3_only",         {}),
        ("02_TD3_LQR",          dict(use_lqr=True, lqr=lqr, alpha=args.alpha)),
        ("03_TD3_LidarAvoid",   dict(use_lidar_avoid=True, lidar_avoider=lidar_avoider)),
        ("04_TD3_MapAvoid",     dict(use_map_avoid=True, map_avoider=map_avoider)),
        ("05_TD3_MapAvoid_LQR", dict(use_map_avoid=True, map_avoider=map_avoider,
                                      use_lqr=True, lqr=lqr, alpha=args.alpha)),
    ]

    if args.only:
        configs = [(n, k) for n, k in configs if n == args.only]
        if not configs:
            print(f"[error] config '{args.only}' not found")
            sys.exit(1)

    for name, kw in configs:
        print(f"\n[record] {name}")
        # Each run gets its own GUI env -- PyBullet GUI is global, so we
        # create + close per run.
        env = HuskyObstacleDeterministicEnv(
            render_mode="human",
            goal_distance=args.goal_distance,
            obstacle_x=args.obstacle_x,
            obstacle_y_offset=args.obstacle_y_offset,
            obstacle_radius=args.obstacle_radius,
        )
        out_path = str(out_dir / f"{name}.mp4")
        run_with_recording(
            env, model, seed=args.seed, record_path=out_path,
            realtime_factor=args.realtime_factor,
            follow_camera=not args.top_down,
            **kw,
        )
        env.close()

    print(f"\n[done] All MP4 videos saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
