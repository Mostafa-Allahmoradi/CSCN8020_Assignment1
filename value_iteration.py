import numpy as np
import time

def standard_value_iteration(mdp, log_file=None):
    """
    Standard VI: Uses two arrays (V and V_new) for synchronous updates.
    Logs details to log_file if provided.
    """
    V = np.zeros((mdp.rows, mdp.cols))
    iteration = 0
    start_time = time.time()
    
    if log_file:
        log_file.write(f"--- Starting Standard Value Iteration ---\n")
    
    while True:
        delta = 0
        V_new = np.copy(V) # Copy prevents in-place updates during sweep
        
        for r in range(mdp.rows):
            for c in range(mdp.cols):
                state = (r, c)
                
                if state == mdp.goal_state:
                    continue
                
                values = []
                for action in mdp.actions:
                    next_s = mdp.get_transition(state, action)
                    reward = mdp.get_reward(next_s)
                    values.append(reward + mdp.gamma * V[next_s])
                
                V_new[r, c] = max(values)
                delta = max(delta, abs(V_new[r, c] - V[r, c]))
        
        V = V_new
        iteration += 1
        
        # LOGGING
        if log_file:
            log_file.write(f"Iteration {iteration}: Max Delta = {delta:.6f}\n")
            log_file.write(f"Grid Values:\n{np.round(V, 3)}\n")
            log_file.write("-" * 30 + "\n")

        if delta < mdp.theta:
            break
            
    end_time = time.time()
    return V, iteration, end_time - start_time

def inplace_value_iteration(mdp, log_file=None):
    """
    In-Place VI: Uses a single array V. Updates are asynchronous.
    Logs details to log_file if provided.
    """
    V = np.zeros((mdp.rows, mdp.cols))
    iteration = 0
    start_time = time.time()
    
    if log_file:
        log_file.write(f"\n--- Starting In-Place Value Iteration ---\n")
    
    while True:
        delta = 0
        
        for r in range(mdp.rows):
            for c in range(mdp.cols):
                state = (r, c)
                
                if state == mdp.goal_state:
                    continue
                
                v_old = V[r, c]
                values = []
                for action in mdp.actions:
                    next_s = mdp.get_transition(state, action)
                    reward = mdp.get_reward(next_s)
                    values.append(reward + mdp.gamma * V[next_s])
                
                # Direct update to V
                V[r, c] = max(values)
                delta = max(delta, abs(V[r, c] - v_old))
        
        iteration += 1

        # LOGGING
        if log_file:
            log_file.write(f"Iteration {iteration}: Max Delta = {delta:.6f}\n")
            log_file.write(f"Grid Values:\n{np.round(V, 3)}\n")
            log_file.write("-" * 30 + "\n")

        if delta < mdp.theta:
            break
            
    end_time = time.time()
    return V, iteration, end_time - start_time