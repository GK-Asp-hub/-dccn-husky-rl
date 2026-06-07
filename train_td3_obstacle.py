"""
Обучение TD3 на HuskyObstacleEnv (Stage 2a: статичные препятствия + лидар).

Задача существенно сложнее, чем Stage 1:
- Нужно избегать препятствий, не проходя сквозь них
- 16-мерное lidar-observation расширяет входную размерность до 25
- Плотный штраф за близость к препятствиям → reward'ы отрицательные на старте

Бюджет: 200k шагов (~30-40 минут CPU) — первая попытка. Если не сойдётся,
пересматриваем reward shaping и noise schedule.

Usage:
    python train_td3_obstacle.py
"""

import os
from pathlib import Path

# torch 2.11 Windows workaround: без этого падает с ModuleNotFoundError: No module named 'pwd'
# (torch._inductor зовёт getpass.getuser(), который на Windows идёт через Unix-only pwd).
# Должно стоять ДО импорта stable_baselines3 (который тащит torch).
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path(".torch_cache").resolve()))

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from envs.husky_obstacle_env import HuskyObstacleEnv


# --- Параметры эксперимента ---
RUN_NAME = "husky_obstacle_td3_v3_steppen036"  # v3: step_penalty 0.01→0.36 для проверки implicit-shaping гипотезы (2026-04-22)
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
SEED = 42

LOGS_DIR = Path("runs") / RUN_NAME
MODEL_DIR = Path("models")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_env(seed: int):
    env = HuskyObstacleEnv(render_mode=None)
    env = Monitor(env, filename=str(LOGS_DIR / "monitor"))
    env.reset(seed=seed)
    return env


def main():
    print(f"=== Training run: {RUN_NAME} ===")
    print(f"Env:           HuskyObstacleEnv (25D obs, 2D action, with lidar)")
    print(f"Timesteps:     {TOTAL_TIMESTEPS:,}")
    print(f"Eval every:    {EVAL_FREQ:,} steps, {N_EVAL_EPISODES} episodes")
    print(f"Logs:          {LOGS_DIR}")
    print(f"Model out:     {MODEL_DIR / (RUN_NAME + '.zip')}")

    train_env = make_env(seed=SEED)
    eval_env = make_env(seed=SEED + 1000)

    n_actions = train_env.action_space.shape[0]
    # Чуть выше exploration noise — задача требует больше поисковой активности,
    # чем в пустой арене, чтобы найти обходы препятствий
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions),
    )

    # Гиперпараметры TD3 (§3.4 статьи). Сети актора и критиков 400-300 — дефолт MlpPolicy в SB3.
    model = TD3(
        policy="MlpPolicy",       # актор + два критика, полносвязные 400-300, ReLU, выход tanh
        env=train_env,
        action_noise=action_noise,
        learning_rate=1e-3,       # шаг Adam
        buffer_size=200_000,      # размер replay-буфера (больше — задача сложнее)
        learning_starts=2_000,    # сколько шагов собирать опыт до старта обучения
        batch_size=256,           # размер мини-батча
        gamma=0.99,               # дисконт-фактор
        tau=0.005,                # коэффициент мягкого обновления целевых сетей
        train_freq=(1, "step"),   # одно обновление на каждом шаге среды
        gradient_steps=1,
        verbose=1,
        tensorboard_log=str(LOGS_DIR / "tb"),
        seed=SEED,
    )
    # Отложенное обновление актора (policy_delay=2) и сглаживание целевого действия
    # (target_policy_noise=0.2, target_noise_clip=0.5) берутся из дефолтов SB3 TD3 —
    # это те самые d, sigma, c из §3.4.

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / f"{RUN_NAME}_best"),
        log_path=str(LOGS_DIR / "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    print("\n--- Training ---\n")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, log_interval=10)

    final_path = MODEL_DIR / f"{RUN_NAME}.zip"
    model.save(str(final_path))
    print(f"\nModel saved: {final_path}")

    print("\n--- Final evaluation (10 deterministic episodes) ---")
    rewards = []
    successes = 0
    collisions = 0
    for i in range(10):
        obs, info = eval_env.reset(seed=SEED + 2000 + i)
        ep_reward = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        reason = info.get("reason", "unknown")
        if reason == "goal_reached":
            successes += 1
        elif reason == "collision":
            collisions += 1
        print(f"  Episode {i+1}: reward={ep_reward:8.2f}, reason={reason}, "
              f"n_obs={info.get('n_obstacles', 0)}")

    print(f"\nMean reward:   {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Success rate:  {successes}/10 = {successes*10}%")
    print(f"Collision rate:{collisions}/10 = {collisions*10}%")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
