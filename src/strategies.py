import numpy as np 



class Strategy: 
    def __str__(self): 
        return "generic strategy"
    
    def pick_action(self): 
        return 0 
    
class EpsilonGreedy(Strategy): 
    def __init__(self, epsilon, rng):
        self.rng = rng  
        self.epsilon = epsilon 
    
        
    def pick_action(self, agent):
        if self.rng.random() < self.epsilon: 
            action = self.rng.choice(agent.k)
        else: 
            action = np.argmax(agent.action_values)
        #self.count += 1 
        return action
    
    def __str__(self): 
        return f"Epsilon={self.epsilon:.2f}"
    
    
class Adaptive(Strategy): 
    def __init__(self, epsilon, rng, nrounds):
        self.rng = rng  
        self.epsilons = [1/t**(1./3) for t in range(1, nrounds+1, 1)]
        self.count = 0 
        
    def pick_action(self, agent):
        if self.rng.random() < self.epsilons[self.count]: 
            action = self.rng.choice(agent.k)
        else: 
            action = np.argmax(agent.action_values)
        self.count += 1 
        return action
    
    def __str__(self): 
        return f"Adaptive"
    
class Random(Strategy): 
    def __init__(self, rng):
        self.rng = rng 
        
    def pick_action(self, agent):
        return self.rng.choice(agent.k)
    
    def __str__(self): 
        return f"Random"
     

        
