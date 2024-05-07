import numpy as np 

class Agent: 
    def __init__(self, bandit, strategy, prior, gamma=0.99): 
        self.k = bandit.k 
        self.strategy = strategy 
        self.prior = prior 
        self.action_values = prior * np.ones(self.k)
        self.action_attempts = np.zeros(self.k)
        self.step = 0 
        self.last_action = None 
        self.gamma = gamma
    
    def __str__(self): 
        return f"{str(self.strategy)}"
    
    def reset(self): 
        self.action_values = self.prior*np.ones(self.k) 
        self.action_attempts = np.zeros(self.k)
        self.step = 0 
        self.last_action = None 
    
    def pick_action(self): 
        action = self.strategy.pick_action(self)
        self.last_action = action 
        return action 
    
    def observe(self, reward): 
        self.action_attempts[self.last_action] += 1 
        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        q = self.action_values[self.last_action]
        
        self.action_values[self.last_action] += g*(reward-q)
        self.step += 1 