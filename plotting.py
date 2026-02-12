import matplotlib.pyplot as plt
import numpy as np

def plot_results(v_hist, policy, Q, env):
    """
    Visualizes:
    1. Training curve (Average V evolution).
    2. The final Policy Grid with Arrows and State Values.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Average State Value Evolution ---
    ax[0].plot(v_hist, color='purple', linewidth=2)
    ax[0].set_title('Avg State Value (V) Evolution')
    ax[0].set_xlabel('Episodes (x100)')
    ax[0].set_ylabel('Avg V(s)')
    ax[0].grid(True, alpha=0.3)
    
    # --- Plot 2: Policy Grid with Values ---
    ax[1].set_xlim(0, 4); ax[1].set_ylim(0, 4)
    ax[1].set_xticks(np.arange(0, 5, 1)); ax[1].set_yticks(np.arange(0, 5, 1))
    ax[1].grid(True, color='gray', linewidth=1)
    
    # Arrow definitions for Up, Down, Left, Right
    arrows = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            
            if state == env.goal_node:
                ax[1].text(c+0.5, 3-r+0.5, 'GOAL', ha='center', va='center', 
                           color='green', fontweight='bold')
                continue
            elif state == env.trap_node:
                ax[1].text(c+0.5, 3-r+0.5, 'TRAP', ha='center', va='center', 
                           color='red', fontweight='bold')
                continue
            
            # Draw Policy Arrow
            best_action = policy[state]
            dx, dy = arrows[best_action]
            
            if state == env.start_node:
                ax[1].text(c+0.2, 3-r+0.8, 'START', ha='left', va='center', 
                           color='blue', fontsize=8, fontweight='bold')

            ax[1].arrow(c+0.5, 3-r+0.5, dx, dy, head_width=0.08, head_length=0.08, 
                        fc='black', ec='black', alpha=0.6)
            
            # Draw State Value V(s) = max Q(s,a)
            qs = [Q[(state, a)] for a in env.action_space]
            v_val = max(qs) if qs else 0.0
            val_color = 'darkgreen' if v_val >= 0 else 'darkred'
            
            ax[1].text(c+0.5, 3-r+0.2, f"{v_val:.1f}", ha='center', va='center', 
                       fontsize=9, color=val_color, fontweight='bold')
            
    ax[1].set_title('Off-Policy Learned Results')
    ax[1].set_aspect('equal')
    plt.tight_layout()
    plt.show()