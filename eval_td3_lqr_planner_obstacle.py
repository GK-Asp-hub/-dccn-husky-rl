"""
Full ablation eval для Stage 2a: HuskyObstacleEnv с препятствиями + лидар.

Запускает 5 режимов на одном и том же наборе seeds:
  1. TD3 baseline (модель husky_obstacle_td3_v1)
  2. TD3 + LQR residual (alpha=0.2)
  3. TD3 + LQR + Planner (N=2)
  4. TD3 + LQR + Planner (N=3)
  5. TD3 + LQR + Planner (N=5)

Метрики: success rate, reward, steps, smoothness, collision rate,
         min_lidar (насколько близко подходил к препятствиям).

Результат сохраняется в JSON для последующей визуализации.

ВНИМАНИЕ: до запуска должно быть завершено train_td3_obstacle.py
и сохранена модель в models/husky_obstacle_td3_v1_best/best_model.zip.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# torch 2.11 Windows workaround — ОБЯЗАТЕЛЬНО до импорта sb3
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import numpy as np
from stable_baselines3 import TD3

from envs.husky_goal_planned_env import HuskyGoalPlannedEnv
from envs.husky_obstacle_env import HuskyObstacleEnv
from controllers.lqr_diff_drive import LQRDiffDriveLateral, combine_actions


# ---------------------------------------------------------------------------
#   Helpers
# ---------------------------------------------------------------------------

def _compute_smoothness(actions: np.ndarray) -> float:
    if len(actions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())


def _episode_summary(
    info: dict, ep_reward: float, steps: int, start_dist: float,
    actions: np.ndarray, min_lidar_seen: float, extra: dict | None = None,
) -> dict:
    res = {
        "steps": steps,
        "reward": ep_reward,
        "reason": info.get("reason", "ongoing"),
        "final_distance": float(info.get("distance", -1.0)),
        "start_distance": start_dist,
        "goal": info["goal"].tolist() if hasattr(info["goal"], "tolist") else list(info["goal"]),
        "smoothness": _compute_smoothness(actions),
        "success": info.get("reason") == "goal_reached",
        "collision": info.get("reason") == "collision",
        "min_lidar_seen": min_lidar_seen,
        "n_obstacles": int(info.get("n_obstacles", 0)),
    }
    if extra:
        res.update(extra)
    return res


# ---------------------------------------------------------------------------
#   Episode runners
# ---------------------------------------------------------------------------

def run_pure_td3(env: HuskyObstacleEnv, model: TD3, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    start_dist = float(np.hypot(info["goal"][0], info["goal"][1]))
    ep_reward, steps = 0.0, 0
    actions = []
    min_lidar = float("inf")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        if "min_lidar" in info:
            min_lidar = min(min_lidar, info["min_lidar"])
        if terminated or truncated:
            break
    return _episode_summary(info, ep_reward, steps, start_dist, np.array(actions), min_lidar)


def run_td3_lqr(env: HuskyObstacleEnv, model: TD3, lqr: LQRDiffDriveLateral,
                alpha: float, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    start_dist = float(np.hypot(info["goal"][0], info["goal"][1]))
    ep_reward, steps = 0.0, 0
    actions = []
    min_lidar = float("inf")
    while True:
        td3_action, _ = model.predict(obs, deterministic=True)
        robot_x, robot_y = float(obs[0]), float(obs[1])
        cos_yaw, sin_yaw = float(obs[5]), float(obs[6])
        robot_yaw = float(np.arctan2(sin_yaw, cos_yaw))
        goal_x = robot_x + float(obs[7])
        goal_y = robot_y + float(obs[8])
        lqr_corr = lqr.compute_correction(robot_x, robot_y, robot_yaw, goal_x, goal_y)
        final_action = combine_actions(td3_action, lqr_corr, alpha=alpha)
        actions.append(final_action.copy())
        obs, reward, terminated, truncated, info = env.step(final_action)
        ep_reward += reward
        steps += 1
        if "min_lidar" in info:
            min_lidar = min(min_lidar, info["min_lidar"])
        if terminated or truncated:
            break
    return _episode_summary(info, ep_reward, steps, start_dist, np.array(actions), min_lidar)


def run_td3_lqr_planner(env: HuskyGoalPlannedEnv, model: TD3,
                        lqr: LQRDiffDriveLateral, alpha: float, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    start_dist = float(np.hypot(info["goal"][0], info["goal"][1]))
    ep_reward, steps = 0.0, 0
    actions = []
    min_lidar = float("inf")
    n_waypoints = int(info["waypoints_all"].shape[0])
    while True:
        td3_action, _ = model.predict(obs, deterministic=True)
        robot_x, robot_y = float(obs[0]), float(obs[1])
        cos_yaw, sin_yaw = float(obs[5]), float(obs[6])
        robot_yaw = float(np.arctan2(sin_yaw, cos_yaw))
        wp_x = robot_x + float(obs[7])
        wp_y = robot_y + float(obs[8])
        lqr_corr = lqr.compute_correction(robot_x, robot_y, robot_yaw, wp_x, wp_y)
        final_action = combine_actions(td3_action, lqr_corr, alpha=alpha)
        actions.append(final_action.copy())
        obs, reward, terminated, truncated, info = env.step(final_action)
        ep_reward += reward
        steps += 1
        if "min_lidar" in info:
            min_lidar = min(min_lidar, info["min_lidar"])
        if terminated or truncated:
            break
    return _episode_summary(
        info, ep_reward, steps, start_dist, np.array(actions), min_lidar,
        extra={"max_wp_idx": int(info["waypoint_idx"]), "n_waypoints": n_waypoints},
    )


# ---------------------------------------------------------------------------
#   Aggregation
# ---------------------------------------------------------------------------

def summarize(label: str, results: list[dict]) -> dict:
    rewards = np.array([r["reward"] for r in results])
    steps = np.array([r["steps"] for r in results])
    smooth = np.array([r["smoothness"] for r in results])
    min_lid = np.array([r["min_lidar_seen"] for r in results])
    success = sum(1 for r in results if r["success"])
    collisions = sum(1 for r in results if r["collision"])
    return {
        "label": label,
        "n_episodes": len(results),
        "success_rate": f"{success}/{len(results)}",
        "collision_rate": f"{collisions}/{len(results)}",
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "steps_mean": float(steps.mean()),
        "steps_std": float(steps.std()),
        "smoothness_mean": float(smooth.mean()),
        "smoothness_std": float(smooth.std()),
        "min_lidar_mean": float(min_lid.mean()),
    }


def print_summary(summ: dict) -> None:
    print(f"\n=== {summ['label']} ({summ['n_episodes']} ep.) ===")
    print(f"  Success:       {summ['success_rate']}")
    print(f"  Collision:     {summ['collision_rate']}")
    print(f"  Reward:        {summ['reward_mean']:+7.2f} +/- {summ['reward_std']:.2f}")
    print(f"  Steps:         {summ['steps_mean']:6.1f} +/- {summ['steps_std']:.1f}")
    print(f"  Smoothness:    {summ['smoothness_mean']:.4f} +/- {summ['smoothness_std']:.4f}")
    print(f"  Min lidar avg: {summ['min_lidar_mean']:.3f} m (чем больше — тем безопаснее маршруты)")


# ---------------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/husky_obstacle_td3_v2_best/best_model.zip",
                        help="v2 — модель после фикса self-hits лидара (2026-04-22). "
                             "Используй v1 только для сравнения старой картины.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=300,
                        help="Разный базовый seed чем в Stage 1 (200), чтобы явно "
                             "показать: это другая среда, другие расположения препятствий.")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--switch-radius", type=float, default=0.8)
    parser.add_argument("--n-list", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--out", type=str, default="ablation_results_obstacle.json")
    args = parser.parse_args()

    assert Path(args.model).exists(), (
        f"Модель не найдена: {args.model}. "
        f"Сначала запусти train_td3_obstacle.py и дождись завершения."
    )

    seeds = [args.seed_base + i for i in range(args.episodes)]
    print(f"Model:         {args.model}")
    print(f"Episodes:      {args.episodes}, seeds {seeds[0]}..{seeds[-1]}")
    print(f"LQR alpha:     {args.alpha}")
    print(f"switch_radius: {args.switch_radius}")
    print(f"N waypoints:   {args.n_list}")
    print()

    all_results: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    # --- 1. Pure TD3 ---
    n_phases = 2 + len(args.n_list)
    print(f">>> 1/{n_phases}  Pure TD3 (obstacle env)")
    env_obs = HuskyObstacleEnv(render_mode=None)
    model = TD3.load(args.model, env=env_obs)
    lqr = LQRDiffDriveLateral()

    results_td3 = []
    for seed in seeds:
        r = run_pure_td3(env_obs, model, seed)
        r["seed"] = seed
        results_td3.append(r)
        print(f"  seed={seed}  steps={r['steps']:3d}  reward={r['reward']:+7.2f}  "
              f"smooth={r['smoothness']:.4f}  min_lidar={r['min_lidar_seen']:.2f}  "
              f"n_obs={r['n_obstacles']}  {r['reason']}")
    all_results["td3"] = results_td3
    summ = summarize("TD3 baseline (obstacles)", results_td3)
    summaries.append(summ)
    print_summary(summ)

    # --- 2. TD3 + LQR ---
    print(f"\n>>> 2/{n_phases}  TD3 + LQR (alpha={args.alpha})")
    results_lqr = []
    for seed in seeds:
        r = run_td3_lqr(env_obs, model, lqr, args.alpha, seed)
        r["seed"] = seed
        results_lqr.append(r)
        print(f"  seed={seed}  steps={r['steps']:3d}  reward={r['reward']:+7.2f}  "
              f"smooth={r['smoothness']:.4f}  min_lidar={r['min_lidar_seen']:.2f}  "
              f"{r['reason']}")
    all_results["td3_lqr"] = results_lqr
    summ = summarize(f"TD3 + LQR (alpha={args.alpha})", results_lqr)
    summaries.append(summ)
    print_summary(summ)

    env_obs.close()

    # --- 3-5. TD3 + LQR + Planner × N ---
    for idx, n_wp in enumerate(args.n_list, start=3):
        print(f"\n>>> {idx}/{n_phases}  TD3 + LQR + Planner (N={n_wp})")
        env_planned = HuskyGoalPlannedEnv(
            render_mode=None, n_waypoints=n_wp, switch_radius=args.switch_radius,
            inner_env_cls=HuskyObstacleEnv,
        )
        model_planned = TD3.load(args.model, env=env_planned)

        results_pl = []
        for seed in seeds:
            r = run_td3_lqr_planner(env_planned, model_planned, lqr, args.alpha, seed)
            r["seed"] = seed
            results_pl.append(r)
            print(f"  seed={seed}  steps={r['steps']:3d}  reward={r['reward']:+7.2f}  "
                  f"smooth={r['smoothness']:.4f}  min_lidar={r['min_lidar_seen']:.2f}  "
                  f"max_wp={r['max_wp_idx']}/{r['n_waypoints']-1}  {r['reason']}")

        all_results[f"td3_lqr_planner_n{n_wp}"] = results_pl
        summ = summarize(f"TD3 + LQR + Planner (N={n_wp}, alpha={args.alpha})", results_pl)
        summaries.append(summ)
        print_summary(summ)

        env_planned.close()

    # --- Итоговая таблица ---
    print("\n" + "=" * 90)
    print("ABLATION SUMMARY (Stage 2a: obstacles + lidar)".center(90))
    print("=" * 90)
    hdr = (
        f"{'Method':<42}"
        f"{'Success':>10}"
        f"{'Collis.':>10}"
        f"{'Steps':>10}"
        f"{'Smooth':>10}"
        f"{'MinLidar':>10}"
    )
    print(hdr)
    print("-" * 90)
    for s in summaries:
        row = (
            f"{s['label']:<42}"
            f"{s['success_rate']:>10}"
            f"{s['collision_rate']:>10}"
            f"{s['steps_mean']:>7.1f}+-{s['steps_std']:>3.0f}"
            f"{s['smoothness_mean']:>7.3f}  "
            f"{s['min_lidar_mean']:>7.2f}m"
        )
        print(row)
    print("=" * 90)

    # --- Сохранение ---
    payload = {
        "config": {
            "model": args.model,
            "seeds": seeds,
            "alpha": args.alpha,
            "switch_radius": args.switch_radius,
            "n_list": args.n_list,
            "stage": "2a (obstacles + lidar)",
        },
        "summaries": summaries,
        "episodes": all_results,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved detailed results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
