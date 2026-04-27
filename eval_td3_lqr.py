"""
Eval: TD3 + LQR Residual Policy на HuskyGoalEnv.

Для каждого сида прогоняет 2 варианта:
  1. Pure TD3 (обученная модель)
  2. TD3 + LQR (residual policy)

Сравнивает: success rate, reward, steps, action smoothness.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# torch 2.11 Windows workaround
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import numpy as np
from stable_baselines3 import TD3

from envs.husky_goal_env import HuskyGoalEnv
from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions


def run_episode_pure_td3(env: HuskyGoalEnv, model: TD3, seed: int):
    obs, info = env.reset(seed=seed)
    start_dist = float(np.hypot(info["goal"][0], info["goal"][1]))
    ep_reward = 0.0
    steps = 0
    actions = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        if terminated or truncated:
            break
    actions = np.array(actions)
    smoothness = float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean()) if len(actions) > 1 else 0.0
    return {
        "steps": steps,
        "reward": ep_reward,
        "reason": info.get("reason", "unknown"),
        "final_distance": info.get("distance", -1.0),
        "goal": info["goal"],
        "start_distance": start_dist,
        "action_smoothness": smoothness,
    }


def run_episode_td3_plus_lqr(
    env: HuskyGoalEnv,
    model: TD3,
    lqr: LQRDiffDriveLateral,
    alpha: float,
    seed: int,
):
    obs, info = env.reset(seed=seed)
    start_dist = float(np.hypot(info["goal"][0], info["goal"][1]))
    ep_reward = 0.0
    steps = 0
    actions = []
    while True:
        # TD3 action
        td3_action, _ = model.predict(obs, deterministic=True)

        # Выдёргиваем состояние из observation:
        #   obs = [x, y, vx, vy, wz, cos_yaw, sin_yaw, goal_dx, goal_dy]
        robot_x, robot_y = float(obs[0]), float(obs[1])
        cos_yaw, sin_yaw = float(obs[5]), float(obs[6])
        robot_yaw = float(np.arctan2(sin_yaw, cos_yaw))
        goal_x = robot_x + float(obs[7])
        goal_y = robot_y + float(obs[8])

        # LQR correction
        lqr_corr = lqr.compute_correction(robot_x, robot_y, robot_yaw, goal_x, goal_y)

        # Combine
        final_action = combine_actions(td3_action, lqr_corr, alpha=alpha)
        actions.append(final_action.copy())

        obs, reward, terminated, truncated, info = env.step(final_action)
        ep_reward += reward
        steps += 1
        if terminated or truncated:
            break
    actions = np.array(actions)
    smoothness = float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean()) if len(actions) > 1 else 0.0
    return {
        "steps": steps,
        "reward": ep_reward,
        "reason": info.get("reason", "unknown"),
        "final_distance": info.get("distance", -1.0),
        "goal": info["goal"],
        "start_distance": start_dist,
        "action_smoothness": smoothness,
    }


def summarize(name: str, results):
    rewards = np.array([r["reward"] for r in results])
    steps = np.array([r["steps"] for r in results])
    smoothness = np.array([r["action_smoothness"] for r in results])
    success = sum(1 for r in results if r["reason"] == "goal_reached")
    print(f"\n=== {name} ===")
    print(f"Success rate:        {success}/{len(results)}")
    print(f"Reward  mean +/- std: {rewards.mean():+.2f} +/- {rewards.std():.2f}")
    print(f"Steps   mean +/- std: {steps.mean():.1f} +/- {steps.std():.1f}")
    print(f"Smooth. mean +/- std: {smoothness.mean():.4f} +/- {smoothness.std():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/husky_td3_v1_cont_best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5, help="LQR residual weight")
    args = parser.parse_args()

    assert Path(args.model).exists(), f"Не найдена модель: {args.model}"

    env = HuskyGoalEnv(render_mode=None)
    model = TD3.load(args.model, env=env)
    lqr = LQRDiffDriveLateral()

    print(f"Model:    {args.model}")
    print(f"Episodes: {args.episodes}, seeds {args.seed_base}..{args.seed_base + args.episodes - 1}")
    print(f"LQR alpha (residual weight): {args.alpha}")

    # Pure TD3
    print("\n--- Pure TD3 ---")
    pure_results = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        r = run_episode_pure_td3(env, model, seed)
        pure_results.append(r)
        print(
            f"Ep {i+1}: seed={seed}  start={r['start_distance']:.2f}m  "
            f"steps={r['steps']:3d}  reward={r['reward']:+8.2f}  "
            f"final={r['final_distance']:.2f}m  smooth={r['action_smoothness']:.4f}  "
            f"{r['reason']}"
        )

    # TD3 + LQR
    print("\n--- TD3 + LQR ---")
    lqr_results = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        r = run_episode_td3_plus_lqr(env, model, lqr, args.alpha, seed)
        lqr_results.append(r)
        print(
            f"Ep {i+1}: seed={seed}  start={r['start_distance']:.2f}m  "
            f"steps={r['steps']:3d}  reward={r['reward']:+8.2f}  "
            f"final={r['final_distance']:.2f}m  smooth={r['action_smoothness']:.4f}  "
            f"{r['reason']}"
        )

    env.close()

    summarize("TD3 only", pure_results)
    summarize("TD3 + LQR", lqr_results)


if __name__ == "__main__":
    main()
