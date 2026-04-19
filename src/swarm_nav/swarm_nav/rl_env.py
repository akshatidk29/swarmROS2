#!/usr/bin/env python3
import random
import numpy as np
import gymnasium as gym

# Abstract RL Training Environment matching the sensor inputs
NUM_ROBOTS = 3
NUM_ACTIONS = 3

ACT_FORWARD = 0
ACT_LEFT = 1
ACT_RIGHT = 2

class RLSensorEnv(gym.Env):
    def __init__(self):
        super(RLSensorEnv, self).__init__()
        # Actions: 0=FWD, 1=LEFT, 2=RIGHT
        self.action_space = gym.spaces.Discrete(3)
        # Observation space matching: 
        # (carrying[4], target_type[4], target_dir[3], wall_front[2], ws_left[2], ws_right[2], visited_ahead[2], last_action[3])
        self.observation_space = gym.spaces.MultiDiscrete([4, 4, 3, 2, 2, 2, 2, 3])
        
        self.step_count = 0
        self.max_steps = 200
        self.state = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = np.array([
            random.randint(0, 3), # carrying
            random.randint(0, 3), # target_type
            random.randint(0, 2), # target_dir
            0, # wall_front
            0, # ws_left
            0, # ws_right
            0, # visited_ahead
            0  # last_action (start moving fwd)
        ])
        return self.state, {}
        
    def step(self, action):
        self.step_count += 1
        carrying, tt, td, wf, wl, wr, va, last_act = self.state
        r = 0.0
        
        # Penalize time passing to encourage efficiency
        r -= 0.1
        
        # Logical transitions and rewards
        if wf == 1 and action == ACT_FORWARD:
            r -= 10.0  # Hitting a wall is terrible
            
        # Avoid visited paths unless surrounded by obstacles
        surrounded = (wf == 1 and wl == 1 and wr == 1)
        if va == 1 and action == ACT_FORWARD:
            if surrounded:
                r += 0.5  # Valid escape route
            else:
                r -= 3.0  # Avoid retracing if not cornered

        # Penalize random rotations (e.g. turning left then right, or spinning)
        if action in [ACT_LEFT, ACT_RIGHT]:
            if action != last_act and last_act in [ACT_LEFT, ACT_RIGHT]:
                r -= 2.0 # Wiggle penalty
            elif action == last_act:
                r -= 0.5 # Spinning penalty, we want to move straight
                
        # Target seeking behavior (Mimics camera/shared map precedence)
        if tt in [1, 2, 3]:
            if td == 0:
                if action == ACT_FORWARD:
                    r += 5.0
                    tt = 0 # Reached target
                else:
                    r -= 2.0 # Ignored target
            elif td == 1: # Target left
                if action == ACT_LEFT:
                    r += 3.0
                    td = 0 # Centered target
                else:
                    r -= 2.0
            elif td == 2: # Target right
                if action == ACT_RIGHT:
                    r += 3.0
                    td = 0 # Centered target
                else:
                    r -= 2.0
        
        # Obstacle avoidance behavior (Mimics wall precedence)
        if wf == 1 and tt == 0:
            if wl == 1 and action == ACT_RIGHT:
                r += 2.0 # Correctly avoid left wall
            elif wr == 1 and action == ACT_LEFT:
                r += 2.0 # Correctly avoid right wall
            elif action in [ACT_LEFT, ACT_RIGHT]:
                r += 1.0 # Good to turn away from front wall

        # Exploration behavior (Move straight in open space)
        if wf == 0 and va == 0 and tt == 0:
            if action == ACT_FORWARD:
                r += 1.0  # Reward moving straight
                # Randomly encounter things during exploration
                if random.random() < 0.2:
                    tt = random.randint(1, 3)
                    td = random.randint(0, 2)
                if random.random() < 0.1:
                    wf = 1
                if random.random() < 0.1:
                    va = 1
                if random.random() < 0.1:
                    wl = 1
                if random.random() < 0.1:
                    wr = 1
            else:
                r -= 1.0  # Penalize turning when there's nothing in front
                
        # Turning clears the path in front
        if action in [ACT_LEFT, ACT_RIGHT]:
            wf = 0
            va = 0
            # Side walls shift depending on turn
            if action == ACT_LEFT:
                wr = wf
                wl = 0
            elif action == ACT_RIGHT:
                wl = wf
                wr = 0
            
        self.state = np.array([carrying, tt, td, wf, wl, wr, va, action])
        done = self.step_count >= self.max_steps
        
        return self.state, r, done, False, {}
