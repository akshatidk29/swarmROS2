#!/usr/bin/env python3
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from rl_env import RLSensorEnv

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_count = 0
        self.rewards = []
        
    def _on_step(self) -> bool:
        if "dones" in self.locals:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    if "episode" in info:
                        self.rewards.append(info["episode"]["r"])
                        self.episode_count += 1
                        
                        if self.episode_count % 1000 == 0:
                            avg_reward = np.mean(self.rewards[-1000:])
                            print(f"--- Episode {self.episode_count} ---")
                            print(f"Average Reward (last 1000 episodes): {avg_reward:.2f}")
        return True

def main():
    print("Initializing RL Sensor Environment...")
    # Wrap the environment so it can be used by Stable Baselines3
    # Use Monitor to automatically add 'episode' info dict for the callback
    from stable_baselines3.common.monitor import Monitor
    env = make_vec_env(lambda: Monitor(RLSensorEnv()), n_envs=4)

    print("Creating PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001, n_steps=2048, batch_size=64)
    
    print("Training PPO Agent for 5,000,000 timesteps...")
    callback = RewardCallback()
    model.learn(total_timesteps=100000, callback=callback)

    # Save the trained model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'ppo_nav_model.zip')
    model.save(model_path)
    print(f"Saved PPO model to {model_path}")

if __name__ == '__main__':
    main()
