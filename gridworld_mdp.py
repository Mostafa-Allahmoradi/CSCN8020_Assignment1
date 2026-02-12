import numpy as np

class GridWorldMDP:
    def __init__(self, gamma=0.9, theta=1e-4):
        self.rows = 5
        self.cols = 5
        self.gamma = gamma
        self.theta = theta
        
        self.goal_state = (4, 4)
        # Grey states (valid but penalized)
        self.grey_states = [(2, 2), (3, 0), (0, 4)]
        
        # Actions
        self.actions = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        self.action_vectors = {
            'RIGHT': (0, 1),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'UP': (-1, 0)
        }

    def is_valid(self, r, c):
        """Check if coordinates are within the grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_reward(self, state):
        """Return R(s) based on current state type."""
        if state == self.goal_state:
            return 10
        elif state in self.grey_states:
            return -5
        else:
            return -1

    def get_transition(self, state, action):
        """Deterministic transition: returns next state."""
        r, c = state
        dr, dc = self.action_vectors[action]
        next_r, next_c = r + dr, c + dc
        
        # If move is valid, return new state; otherwise stay put
        if self.is_valid(next_r, next_c):
            return (next_r, next_c)
        else:
            return state

    def get_optimal_policy(self, V):
        """Derive the optimal policy pi* from the value function V*."""
        policy = np.empty((self.rows, self.cols), dtype=object)
        
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                
                # Terminal state has no action
                if state == self.goal_state:
                    policy[r, c] = 'G'
                    continue
                
                best_val = -float('inf')
                best_action = None
                
                for action in self.actions:
                    next_s = self.get_transition(state, action)
                    reward = self.get_reward(next_s)
                    
                    # Bellman expectation equation
                    val = reward + self.gamma * V[next_s]
                    
                    if val > best_val:
                        best_val = val
                        best_action = action
                
                # Visual arrows for policy
                arrow_map = {'RIGHT': '→', 'DOWN': '↓', 'LEFT': '←', 'UP': '↑'}
                policy[r, c] = arrow_map[best_action] if best_action else ' '
                
        return policy