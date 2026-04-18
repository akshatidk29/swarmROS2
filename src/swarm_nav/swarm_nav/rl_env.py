#!/usr/bin/env python3
import random
import numpy as np

# Abstract RL Training Environment matching the sensor inputs

NUM_ROBOTS = 3
NUM_ACTIONS = 3

ACT_FORWARD = 0
ACT_LEFT = 1
ACT_RIGHT = 2

class RLSensorEnv:
    def __init__(self):
        # We simulate a purely abstract scenario for training the Q-table
        # Because we only need the Q-table to map sensor inputs to actions.
        self.step_count = 0
        self.max_steps = 200
        
    def reset(self):
        self.step_count = 0
        
    def get_state_keys(self):
        # Return 3 random valid states for training exploration
        return [self._random_state() for _ in range(NUM_ROBOTS)]
        
    def _random_state(self):
        carrying = random.randint(0, 3)
        target_type = random.randint(0, 3)
        target_dir = random.randint(0, 2)
        wall_front = random.randint(0, 1)
        visited_ahead = random.randint(0, 1)
        return (carrying, target_type, target_dir, wall_front, visited_ahead)
        
    def step(self, states, actions):
        self.step_count += 1
        rewards = []
        for s, a in zip(states, actions):
            carrying, tt, td, wf, va = s
            r = -0.1  # base time penalty
            
            # Logic:
            if wf == 1 and a == ACT_FORWARD:
                r -= 5.0  # hitting wall
            
            if va == 1 and a == ACT_FORWARD:
                r -= 2.0  # retracing
                
            if tt == 1 or tt == 2:  # we see something we want
                if td == 0 and a == ACT_FORWARD:
                    r += 2.0  # moving towards it
                elif td == 1 and a == ACT_LEFT:
                    r += 1.0  # turning towards it
                elif td == 2 and a == ACT_RIGHT:
                    r += 1.0  # turning towards it
                else:
                    r -= 1.0  # ignoring target
            
            if wf == 0 and va == 0 and tt == 0 and a == ACT_FORWARD:
                r += 0.5  # exploring new areas
                
            rewards.append(r)
            
        done = self.step_count >= self.max_steps
        return rewards, done

