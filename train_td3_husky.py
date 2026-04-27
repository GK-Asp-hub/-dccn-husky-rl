"""
Первое обучение TD3 на HuskyGoalEnv.

Цель этого run: проверить, что агент вообще учится на нашей задаче.
Не полный эксперимент — короткий пилот на 50k шагов с eval каждые 5k.

Если кривая обучения идёт вверх и eval_reward становится положительным —
можно будет запускать полноценный эксперимент для статьи.

Usage:
    python train_td3_husky.py
"""

from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from envs.husky_goal_env import HuskyGoalEnv


# --- Параметры эксперимента ---
RUN_NAME = "husky_td3_v1"
TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 5
SEED = 42

LOGS_DIR = Path("runs") / RUN_NAME
MODEL_DIR = Path("models")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_env(seed: int):
    env = HuskyGoalEnv(render_mode=None)
    env = Monitor(env, filename=str(LOGS_DIR / "monitor"))
    env.reset(seed=seed)
    return env


def main():
    print(f"=== Training run: {RUN_NAME} ===")
    print(f"Timesteps:     {TOTAL_TIMESTEPS:,}")
    print(f"Eval every:    {EVAL_FREQ:,} steps, {N_EVAL_EPISODES} episodes")
    print(f"Logs:          {LOGS_DIR}")
    print(f"Model out:     {MODEL_DIR / (RUN_NAME + '.zip')}")

    train_env = make_env(seed=SEED)
    eval_env = make_env(seed=SEED + 1000)

    n_actions = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions),
    )

    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        action_noise=action_noise,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
        tensorboard_log=str(LOGS_DIR / "tb"),
        seed=SEED,
    )

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
        print(f"  Episode {i+1}: reward={ep_reward:7.2f}, reason={reason}")

    print(f"\nMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Success rate: {successes}/10 = {successes*10}%")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
