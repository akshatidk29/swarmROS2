#!/usr/bin/env python3
"""
Fine-tune a pretrained PPO model using Gazebo-sourced observations.

This bridges the domain gap between the Pygame training env and Gazebo.
Loads the pretrained policy and continues training with Gazebo inputs.

Usage:
    # Start Gazebo sim first, then:
    python3 finetune.py --timesteps 50000
    python3 finetune.py --timesteps 100000 --render
    python3 finetune.py --timesteps 50000 --headless  # no Gazebo GUI (default)

Prerequisites:
    1. Gazebo simulation running with robots spawned
    2. Pre-trained model at RL/model/policy/policy.zip
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.config import OBS_DIM, N_STEPS_PPO, MAX_STEPS

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
DEFAULT_MODEL = os.path.join(_ROOT, 'RL', 'model', 'policy', 'policy.zip')
FINETUNE_LOG_DIR = os.path.join(_HERE, 'logs')


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune RL model on Gazebo observations')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Path to pretrained model .zip')
    parser.add_argument('--timesteps', type=int, default=50_000,
                        help='Total fine-tuning timesteps')
    parser.add_argument('--robot-ns', type=str, default='robot_1',
                        help='ROS2 namespace of robot to control')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (lower than pretraining)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Where to save fine-tuned model')
    parser.add_argument('--render', action='store_true',
                        help='Enable Gazebo GUI rendering (gzclient)')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run without Gazebo GUI (default)')
    parser.add_argument('--step-duration', type=float, default=0.1,
                        help='Seconds per env step')
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS,
                        help='Max steps per episode')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.render:
        args.headless = False
        print('Gazebo GUI rendering enabled')
    elif args.headless:
        print('Running headless (no Gazebo GUI)')
        os.environ['GAZEBO_HEADLESS'] = '1'

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from gazebo_env_wrapper import GazeboEnvWrapper

    # Setup
    save_dir = args.save_dir or os.path.join(_ROOT, 'RL', 'model', 'policy')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(FINETUNE_LOG_DIR, exist_ok=True)

    run_name = f'ft_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = os.path.join(FINETUNE_LOG_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f'═══ Gazebo Fine-Tuning ═══')
    print(f'  Model      : {args.model}')
    print(f'  Timesteps  : {args.timesteps:,}')
    print(f'  Robot NS   : {args.robot_ns}')
    print(f'  LR         : {args.lr}')
    print(f'  Save dir   : {save_dir}')
    print(f'  Run dir    : {run_dir}')
    print(f'  Headless   : {args.headless}')
    print()

    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    # Create env(s)
    if args.robot_ns.lower() == 'all':
        def make_env(robot_idx):
            def _init():
                return Monitor(GazeboEnvWrapper(
                    robot_ns=f'robot_{robot_idx}',
                    step_duration=args.step_duration,
                    max_steps=args.max_steps,
                    render_mode='human' if args.render else None,
                ))
            return _init
        
        env = SubprocVecEnv([make_env(i) for i in range(1, 5)])
        print(f'Created SubprocVecEnv for 4 robots')
    else:
        env = DummyVecEnv([lambda: Monitor(GazeboEnvWrapper(
            robot_ns=args.robot_ns,
            step_duration=args.step_duration,
            max_steps=args.max_steps,
            render_mode='human' if args.render else None,
        ))])
        print(f'Created DummyVecEnv for {args.robot_ns}')

    # Load pretrained model with new env
    print(f'Loading pretrained model from {args.model}...')
    model = PPO.load(
        args.model,
        env=env,
        learning_rate=args.lr,
        device=args.device,
        tensorboard_log=run_dir,
    )
    print('Model loaded. Starting fine-tuning...')

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=True,
        progress_bar=True,
    )
    train_time = time.time() - t0

    # Save
    ft_path = os.path.join(save_dir, 'policy_finetuned')
    model.save(ft_path)
    print(f'\n✓ Fine-tuned model saved to {ft_path}.zip')

    # Summary
    summary = {
        'base_model': args.model,
        'timesteps': args.timesteps,
        'learning_rate': args.lr,
        'robot_ns': args.robot_ns,
        'train_time_s': round(train_time, 1),
        'obs_dim': OBS_DIM,
        'device': args.device,
        'headless': args.headless,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(run_dir, 'finetune_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n═══ Fine-Tuning Complete ═══')
    print(f'  Time      : {train_time:.0f}s')
    print(f'  Run dir   : {run_dir}')

    env.close()


if __name__ == '__main__':
    main()
