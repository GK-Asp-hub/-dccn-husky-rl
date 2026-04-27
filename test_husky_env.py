"""Smoke test для HuskyGoalEnv: reset, 50 шагов случайной policy, проверка формата."""
import numpy as np
from envs.husky_goal_env import HuskyGoalEnv


def main():
    env = HuskyGoalEnv(render_mode=None)  # headless
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"\nReset returned obs shape={obs.shape}, dtype={obs.dtype}")
    print(f"obs: {obs}")
    print(f"goal: {info['goal']}")

    # Проверка: obs внутри observation_space
    assert env.observation_space.contains(obs), f"Obs out of space: {obs}"

    rewards = []
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            print(f"  Episode ended at step {step+1}: reason={info.get('reason')}, reward={reward:.2f}")
            break
    else:
        print(f"  Прошло 50 шагов без termination, reward последнего = {rewards[-1]:.3f}")

    print(f"\nReward stats over {len(rewards)} steps:")
    print(f"  min={min(rewards):.3f}, max={max(rewards):.3f}, mean={np.mean(rewards):.3f}")
    print(f"  sum={sum(rewards):.3f}")

    env.close()
    print("\nHuskyGoalEnv smoke test OK.")


if __name__ == "__main__":
    main()
