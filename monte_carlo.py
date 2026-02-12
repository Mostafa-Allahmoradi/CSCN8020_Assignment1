import numpy as np
import random
import logging
from collections import defaultdict
from grid_world_mc import GridWorldMC

def monte_carlo_off_policy(episodes=10000, gamma=0.9):
    """
    Performs Off-Policy Monte Carlo Control using Weighted Importance Sampling.
    """
    env = GridWorldMC()
    
    # --- Initialization ---
    Q = defaultdict(float)    
    C = defaultdict(float)    
    
    # Target Policy (pi): Greedy w.r.t Q
    pi = {s: np.random.choice(env.action_space) for s in env.state_space}
    
    # Behavior Policy (b): Uniform Random
    b_prob = 0.25
    
    history_v = []

    # --- Training Loop ---
    for ep in range(episodes):
        
        # 1. Generate Episode
        s0 = random.choice(env.state_space)
        while s0 == env.goal_node or s0 == env.trap_node:
            s0 = random.choice(env.state_space)
            
        episode = []
        state = s0
        episode_reward = 0
        
        # Run episode
        for _ in range(100): 
            action = random.choice(env.action_space)
            next_state, reward, done = env.step(state, action)
            episode.append((state, action, reward))
            episode_reward += reward
            if done: break
            state = next_state
        
        # 2. Policy Evaluation & Improvement
        G = 0.0
        W = 1.0 
        
        for t in range(len(episode)-1, -1, -1):
            s_t, a_t, r_next = episode[t]
            G = gamma * G + r_next
            C[(s_t, a_t)] += W
            
            if C[(s_t, a_t)] != 0:
                Q[(s_t, a_t)] += (W / C[(s_t, a_t)]) * (G - Q[(s_t, a_t)])
            
            q_values = [Q[(s_t, a)] for a in env.action_space]
            best_a = np.argmax(q_values)
            pi[s_t] = best_a
            
            if a_t != best_a:
                break
            
            W = W * (1.0 / b_prob)
            
        # --- Logging ---
        # Log details every 100 episodes to the text file
        if ep % 50 == 0:
            v_avg = np.mean([max([Q[(s, a)] for a in env.action_space]) for s in env.state_space])
            history_v.append(v_avg)
            logging.info(f"Episode {ep}: Avg Value={v_avg:.4f} | Total Reward={episode_reward}")

    return history_v, pi, Q, env