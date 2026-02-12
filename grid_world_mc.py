class GridWorldMC:
    """
    A simple 4x4 Grid Environment.
    Start: (0, 0) -> Starting position with -1 reward per step
    Trap:  (1,1) -> Ends episode with -50 reward
    Goal:  (3,3) -> Ends episode with +10 reward
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    """
    def __init__(self):
        self.rows, self.cols = 4, 4
        self.state_space = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.action_space = [0, 1, 2, 3] 
        self.start_node = (0, 0) 
        self.trap_node = (1, 1)
        self.goal_node = (3, 3)

    def step(self, state, action):
        """
        Takes a step in the environment.
        Returns: next_state, reward, done
        """
        if state == self.goal_node: return state, 0, True
        
        r, c = state
        # Action logic
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(3, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(3, c + 1)
        
        next_state = (r, c)
        
        # Reward Structure
        if next_state == self.goal_node:
            reward = 10
            done = True
        elif next_state == self.trap_node:
            reward = -50
            done = True
        else:
            reward = -1
            done = False
            
        return next_state, reward, done