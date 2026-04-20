#!/usr/bin/env python3
"""
Curriculum-based PPO training for swarm collector.

3 Phases:
  Phase 1 — Pick & Deliver: High pick/deliver rewards, mild collision penalties
  Phase 2 — Collision Avoidance: Full collision penalties activated
  Phase 3 — Full Exploration: Jerk penalty and refined rewards

Each phase continues from the previous phase's model (continual learning).
Lighter network: [128, 64] instead of [256, 256, 128].

Usage:
    python3 train_curriculum.py
    python3 train_curriculum.py --device cpu --n-envs 4
    python3 train_curriculum.py --phase 2 --resume RL/model/policy/phase_1.zip
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

sys.path.insert(0, os.path.dirname(__file__))

# IMPORTANT: import config BEFORE creating envs so we can override rewards
import config
from env.swarm_env import SwarmCollectorEnv
from config import (
    OBS_DIM, ACT_DIM, MAX_STEPS, N_STEPS_PPO, N_ENVS_DEFAULT, CURRICULUM,
)
from paths import next_run_dir, model_path


# ═══════════════ STDOUT TEE ═══════════════

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# ═══════════════ CALLBACK ═══════════════

class PhaseCallback(BaseCallback):
    """Log per-episode metrics during training."""

    def __init__(self, print_freq=10000):
        super().__init__()
        self.episode_rewards = []
        self.pick_count = []
        self.delivery_count = []
        self.collision_count = []
        self.print_freq = print_freq
        self.last_print_steps = 0

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
            if 'picks' in info:
                self.pick_count.append(info.get('picks', 0))
                self.delivery_count.append(info.get('deliveries', 0))
                self.collision_count.append(info.get('collisions', 0))
                
        # Print progress during the phase
        if self.num_timesteps - self.last_print_steps >= self.print_freq:
            self.last_print_steps = self.num_timesteps
            s = self.summary(n_last=20)
            print(f'    [Step {self.num_timesteps:,}] Episodes: {s["episodes"]} | '
                  f'Reward: {s["mean_reward"]:.1f} | '
                  f'Picks: {s["mean_picks"]:.1f} | '
                  f'Deliveries: {s["mean_deliveries"]:.1f} | '
                  f'Collisions: {s["mean_collisions"]:.1f}')
            
        return True

    def summary(self, n_last=50):
        """Get summary of last n episodes."""
        r = self.episode_rewards[-n_last:] if self.episode_rewards else [0]
        p = self.pick_count[-n_last:] if self.pick_count else [0]
        d = self.delivery_count[-n_last:] if self.delivery_count else [0]
        c = self.collision_count[-n_last:] if self.collision_count else [0]
        return {
            'mean_reward': round(float(np.mean(r)), 2),
            'mean_picks': round(float(np.mean(p)), 2),
            'mean_deliveries': round(float(np.mean(d)), 2),
            'mean_collisions': round(float(np.mean(c)), 2),
            'episodes': len(self.episode_rewards),
        }


# ═══════════════ HELPERS ═══════════════

def make_env(rank, seed=0, max_steps=None):
    def _init():
        env = SwarmCollectorEnv(max_steps=max_steps)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def apply_rewards(phase_cfg):
    """Override reward constants in config module for this phase."""
    rewards = phase_cfg.get('rewards', {})
    for key, val in rewards.items():
        if hasattr(config, key):
            setattr(config, key, val)
    print(f'  Applied {len(rewards)} reward overrides')


def save_plots(log_dir, cb, phase_name):
    """Save reward curve for this phase."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not cb.episode_rewards:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{phase_name} Training Curves', fontsize=14)

    data = [
        (cb.episode_rewards, 'Reward', '#3498db'),
        (cb.pick_count, 'Picks/ep', '#27ae60'),
        (cb.delivery_count, 'Deliveries/ep', '#9b59b6'),
        (cb.collision_count, 'Collisions/ep', '#e74c3c'),
    ]

    for ax, (vals, label, color) in zip(axes.flat, data):
        if vals:
            ax.plot(vals, alpha=0.3, color=color)
            if len(vals) > 20:
                w = min(50, len(vals) // 3)
                sm = np.convolve(vals, np.ones(w)/w, mode='valid')
                ax.plot(range(w-1, len(vals)), sm, color=color, lw=2)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(log_dir, f'{phase_name}_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════ MAIN ═══════════════

def main():
    parser = argparse.ArgumentParser(description='Curriculum PPO training')
    parser.add_argument('--n-envs', type=int, default=N_ENVS_DEFAULT)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--phase', type=int, default=None,
                        help='Run only this phase (1, 2, or 3)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from model .zip')
    args = parser.parse_args()

    save_dir = args.save_dir or model_path()
    os.makedirs(save_dir, exist_ok=True)

    run_dir = next_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # Tee stdout
    log_file = open(os.path.join(run_dir, 'output.txt'), 'w', buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f'═══ Curriculum Training ═══')
    print(f'  Obs dim     : {OBS_DIM}')
    print(f'  N envs      : {args.n_envs}')
    print(f'  Device      : {args.device}')
    print(f'  Save dir    : {save_dir}')
    print(f'  Run dir     : {run_dir}')
    print(f'  Network     : pi=[128, 64], vf=[128, 64]')
    print()

    set_random_seed(args.seed)

    # Create envs
    env = SubprocVecEnv([
        make_env(i, seed=args.seed, max_steps=args.max_steps)
        for i in range(args.n_envs)
    ])
    eval_env = DummyVecEnv([
        make_env(0, seed=args.seed + 10_000, max_steps=args.max_steps)
    ])

    # Select phases
    phase_keys = list(CURRICULUM.keys())
    if args.phase is not None:
        phase_keys = [f'phase_{args.phase}']

    model = None
    prev_model_path = args.resume
    all_results = {}

    for phase_key in phase_keys:
        phase_cfg = CURRICULUM[phase_key]
        phase_name = phase_cfg['name']
        phase_steps = phase_cfg['timesteps']

        print(f'\n{"="*60}')
        print(f'  PHASE: {phase_name} ({phase_key})')
        print(f'  Timesteps: {phase_steps:,}')
        print(f'{"="*60}')

        # Apply reward overrides
        apply_rewards(phase_cfg)

        # Create or load model
        if prev_model_path and os.path.exists(prev_model_path):
            print(f'  Loading from: {prev_model_path}')
            model = PPO.load(prev_model_path, env=env, device=args.device)
            # Update learning rate for this phase
            model.learning_rate = args.lr
        elif model is None:
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=args.lr,
                n_steps=N_STEPS_PPO,
                batch_size=max(64, (N_STEPS_PPO * args.n_envs) // 16),
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=0,
                tensorboard_log=run_dir,
                policy_kwargs=dict(
                    net_arch=dict(pi=[128, 64], vf=[128, 64]),
                ),
                device=args.device,
            )

        # Callbacks
        cb = PhaseCallback()
        phase_ckpt_dir = os.path.join(run_dir, phase_key, 'checkpoints')
        os.makedirs(phase_ckpt_dir, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(25_000 // args.n_envs, 1),
            save_path=phase_ckpt_dir,
            name_prefix=phase_key,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(run_dir, phase_key),
            eval_freq=max(25_000 // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            verbose=0,
        )

        # Train
        t0 = time.time()
        model.learn(
            total_timesteps=phase_steps,
            callback=[cb, checkpoint_cb, eval_cb],
            reset_num_timesteps=True,
            tb_log_name=phase_key,
        )
        elapsed = time.time() - t0

        # Save phase model
        phase_save = os.path.join(save_dir, f'{phase_key}.zip')
        model.save(phase_save)
        prev_model_path = phase_save

        # Print summary
        summary = cb.summary()
        summary['elapsed_seconds'] = round(elapsed, 1)
        all_results[phase_key] = summary

        print(f'\n  ── {phase_name} Complete ──')
        print(f'  Time       : {elapsed:.0f}s')
        print(f'  Episodes   : {summary["episodes"]}')
        print(f'  Mean reward: {summary["mean_reward"]}')
        print(f'  Mean picks : {summary["mean_picks"]}')
        print(f'  Mean deliv : {summary["mean_deliveries"]}')
        print(f'  Mean collis: {summary["mean_collisions"]}')
        print(f'  Saved      : {phase_save}')

        # Save plots
        save_plots(run_dir, cb, phase_key)

    # Save final model as policy.zip
    final_path = os.path.join(save_dir, 'policy.zip')
    if model is not None:
        model.save(final_path)
        print(f'\n✓ Final model saved: {final_path}')

    # Save results summary
    results_path = os.path.join(run_dir, 'curriculum_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\n═══ Curriculum Training Complete ═══')
    for pk, res in all_results.items():
        print(f'  {pk}: reward={res["mean_reward"]}, '
              f'picks={res["mean_picks"]}, '
              f'deliveries={res["mean_deliveries"]}, '
              f'collisions={res["mean_collisions"]}')

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
