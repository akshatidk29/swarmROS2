#!/usr/bin/env python3
import os
import pickle
import random
import numpy as np
from rl_env import RLSensorEnv, NUM_ROBOTS, NUM_ACTIONS

EPISODES = 50000
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
DECAY = 0.9999

def main():
    env = RLSensorEnv()
    q_table = {}

    def get_q(s):
        if s not in q_table:
            q_table[s] = np.zeros(NUM_ACTIONS)
        return q_table[s]

    epsilon = EPSILON_START
    
    print("Training RL Sensor Policy...")
    for ep in range(EPISODES):
        env.reset()
        done = False
        states = env.get_state_keys()
        
        while not done:
            actions = []
            for s in states:
                if random.random() < epsilon:
                    actions.append(random.randint(0, NUM_ACTIONS - 1))
                else:
                    actions.append(int(np.argmax(get_q(s))))
                    
            rewards, done = env.step(states, actions)
            next_states = env.get_state_keys()
            
            for i in range(NUM_ROBOTS):
                s = states[i]
                a = actions[i]
                ns = next_states[i]
                old = get_q(s)[a]
                nxt = np.max(get_q(ns))
                get_q(s)[a] = old + ALPHA * (rewards[i] + GAMMA * nxt - old)
                
            states = next_states
            
        epsilon = max(EPSILON_END, epsilon * DECAY)
        if (ep + 1) % 10000 == 0:
            print(f"Episode {ep+1}, Epsilon: {epsilon:.3f}, States: {len(q_table)}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    q_path = os.path.join(base_dir, 'q_table.pkl')
    with open(q_path, 'wb') as f:
        pickle.dump(dict(q_table), f)
    print(f"Saved Q-table to {q_path}")

if __name__ == '__main__':
    main()
