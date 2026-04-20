#!/usr/bin/env python3
import random
import numpy as np
import gymnasium as gym

NUM_ROBOTS = 3
NUM_ACTIONS = 3

ACT_FORWARD = 0
ACT_LEFT = 1
ACT_RIGHT = 2

class RLSensorEnv(gym.Env):
    def __init__(self):
        super(RLSensorEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.MultiDiscrete([4, 4, 3, 2, 2, 2, 2, 3])
        
        self.step_count = 0
        self.max_steps = 200
        self.state = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = np.array([
            random.randint(0, 3),
            random.randint(0, 3),
            random.randint(0, 2),
            0,
            0,
            0,
            0,
            0
        ])
        return self.state, {}
        
    def step(self, action):
        self.step_count += 1
        carrying, tt, td, wf, wl, wr, va, last_act = self.state
        r = 0.0
        
        r -= 0.1
        
        if wf == 1 and action == ACT_FORWARD:
            r -= 10.0
            
        surrounded = (wf == 1 and wl == 1 and wr == 1)
        if va == 1 and action == ACT_FORWARD:
            if surrounded:
                r += 0.5
            else:
                r -= 3.0

        if action in [ACT_LEFT, ACT_RIGHT]:
            if action != last_act and last_act in [ACT_LEFT, ACT_RIGHT]:
                r -= 2.0
            elif action == last_act:
                r -= 0.5
                
        if tt in [1, 2, 3]:
            if td == 0:
                if action == ACT_FORWARD:
                    r += 5.0
                    tt = 0
                else:
                    r -= 2.0
            elif td == 1:
                if action == ACT_LEFT:
                    r += 3.0
                    td = 0
                else:
                    r -= 2.0
            elif td == 2:
                if action == ACT_RIGHT:
                    r += 3.0
                    td = 0
                else:
                    r -= 2.0
        
        if wf == 1 and tt == 0:
            if wl == 1 and action == ACT_RIGHT:
                r += 2.0
            elif wr == 1 and action == ACT_LEFT:
                r += 2.0
            elif action in [ACT_LEFT, ACT_RIGHT]:
                r += 1.0

        if wf == 0 and va == 0 and tt == 0:
            if action == ACT_FORWARD:
                r += 1.0
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
                r -= 1.0
                
        if action in [ACT_LEFT, ACT_RIGHT]:
            wf = 0
            va = 0
            if action == ACT_LEFT:
                wr = wf
                wl = 0
            elif action == ACT_RIGHT:
                wl = wf
                wr = 0
            
        self.state = np.array([carrying, tt, td, wf, wl, wr, va, action])
        done = self.step_count >= self.max_steps
        
        return self.state, r, done, False, {}
