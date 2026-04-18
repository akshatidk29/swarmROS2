#!/usr/bin/env python3
"""
PPO training for the swarm collector.

Usage:
    python3 train.py --timesteps 1000000 --render
    python3 train.py --timesteps 500000 --save-dir model/train1/
"""

import argparse
import os
import json
import time
from datetime import datetime

import numpy as np

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.insert(0, os.path.dirname(__file__))
from env.swarm_env import SwarmCollectorEnv
from config import MAX_STEPS, N_ROBOTS, DT


class RewardLoggerCallback(BaseCallback):
    """Log episode rewards for plotting."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        return True


def make_env(rank, seed=0, render=False, max_steps=None):
    def _init():
        env = SwarmCollectorEnv(
            render_mode='human' if render else None,
            max_steps=max_steps,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def save_plots(log_dir, callback):
    """Save reward and loss plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    rewards = callback.episode_rewards
    if not rewards:
        return

    # Reward curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color='#3498db', label='Episode reward')
    if len(rewards) > 50:
        window = min(100, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed,
                color='#e74c3c', lw=2, label=f'Moving avg ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(log_dir, 'reward_curve.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # Episode length
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.episode_lengths, alpha=0.3, color='#2ecc71')
    if len(callback.episode_lengths) > 50:
        window = min(100, len(callback.episode_lengths) // 5)
        smoothed = np.convolve(callback.episode_lengths,
                               np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(callback.episode_lengths)), smoothed,
                color='#e74c3c', lw=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(log_dir, 'episode_length.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print(f"Plots saved to {log_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train swarm collector PPO')
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--save-dir', type=str, default='model/train1/')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS,
                        help='Max steps per robot before truncation')
    parser.add_argument('--eval-freq', type=int, default=25_000,
                        help='Eval frequency in timesteps')
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--checkpoint-freq', type=int, default=50_000,
                        help='Checkpoint frequency in timesteps')
    parser.add_argument('--early-stop-evals', type=int, default=10,
                        help='Stop after N evals without improvement (0 to disable)')
    parser.add_argument('--min-evals', type=int, default=5,
                        help='Minimum evals before early-stop checks')
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to .zip to resume from')
    args = parser.parse_args()

    # Directories
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', f'run_{run_id}')
    os.makedirs(log_dir, exist_ok=True)

    print(f"═══ Swarm Collector PPO Training ═══")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  N envs    : {args.n_envs}")
    print(f"  Save dir  : {args.save_dir}")
    print(f"  Log dir   : {log_dir}")
    print(f"  Render    : {args.render}")
    print(f"  Max steps : {args.max_steps} per robot")
    print(f"  Max policy steps/episode : {args.max_steps * N_ROBOTS}")
    print(f"  Seed      : {args.seed}")
    print(f"  Device    : {args.device}")

    set_random_seed(args.seed)

    # Create envs
    if args.render:
        args.n_envs = 1
        env = DummyVecEnv([
            make_env(0, seed=args.seed, render=True, max_steps=args.max_steps)
        ])
    else:
        env = SubprocVecEnv([
            make_env(i, seed=args.seed, max_steps=args.max_steps)
            for i in range(args.n_envs)
        ])

    # Eval env
    eval_env = DummyVecEnv([
        make_env(0, seed=args.seed + 10_000, max_steps=args.max_steps)
    ])

    # Callbacks
    reward_cb = RewardLoggerCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=args.save_dir,
        name_prefix='ppo_swarm',
    )
    stop_cb = None
    if args.early_stop_evals and args.early_stop_evals > 0:
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.early_stop_evals,
            min_evals=args.min_evals,
            verbose=1,
        )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
        callback_after_eval=stop_cb,
    )

    # Model
    if args.resume:
        print(f"  Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            device=args.device,
        )

    # Train
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[reward_cb, checkpoint_cb, eval_cb],
        log_interval=args.log_interval,
        progress_bar=True,
    )
    train_time = time.time() - t0

    # Save final model
    final_path = os.path.join(args.save_dir, 'policy')
    model.save(final_path)
    print(f"\n✓ Model saved to {final_path}.zip")

    # Save training summary
    episode_len_mean = (float(np.mean(reward_cb.episode_lengths))
                        if reward_cb.episode_lengths else 0.0)
    episode_robot_steps_mean = episode_len_mean / N_ROBOTS if N_ROBOTS else 0.0
    episode_sim_seconds_mean = episode_robot_steps_mean * DT

    eval_best_success = getattr(eval_cb, 'best_success_rate', None)
    eval_best_mean = getattr(eval_cb, 'best_mean_reward', None)

    summary = {
        'timesteps': args.timesteps,
        'train_time_s': round(train_time, 1),
        'n_episodes': len(reward_cb.episode_rewards),
        'final_mean_reward': round(float(np.mean(reward_cb.episode_rewards[-50:]))
                                   if reward_cb.episode_rewards else 0, 2),
        'best_reward': round(float(max(reward_cb.episode_rewards))
                             if reward_cb.episode_rewards else 0, 2),
        'episode_length_policy_steps_mean': round(episode_len_mean, 1),
        'episode_length_robot_steps_mean': round(episode_robot_steps_mean, 1),
        'episode_length_sim_seconds_mean': round(episode_sim_seconds_mean, 1),
        'max_steps_per_robot': args.max_steps,
        'max_steps_policy': args.max_steps * N_ROBOTS,
        'dt': DT,
        'eval_best_mean_reward': (
            round(float(eval_best_mean), 2) if eval_best_mean is not None else None
        ),
        'eval_best_success_rate': (
            round(float(eval_best_success), 3)
            if eval_best_success is not None else None
        ),
        'lr': args.lr,
        'n_envs': args.n_envs,
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'device': args.device,
    }
    with open(os.path.join(log_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save plots
    save_plots(log_dir, reward_cb)

    print(f"\n═══ Training Complete ═══")
    print(f"  Time     : {train_time:.0f}s")
    print(f"  Episodes : {summary['n_episodes']}")
    print(f"  Mean R   : {summary['final_mean_reward']}")
    print(f"  Best R   : {summary['best_reward']}")
    print(f"\n  To deploy: cp {final_path}.zip ../RL/model/policy.zip")

    if summary['n_episodes'] == 0:
        max_policy_steps = args.max_steps * N_ROBOTS
        print(
            "\n  Note: No episodes finished. This is expected if total "
            f"timesteps < {max_policy_steps} (max steps per episode)."
        )

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
