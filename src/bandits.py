import numpy as np 




class MultiArmedBandit: 
    def __init__(self, k): 
        self.k = k 
        self.action_values = np.zeros(self.k)
        self.optimal = 0 
        
    def reset(self): 
        self.action_values = np.zeros(self.k)
        self.optimal = 0 
        
    def pull(self, action): 
        return 0, action
    
class GaussianBandit(MultiArmedBandit): 
    def __init__(self, k, rng, mu=0, sigma=1): 
        super(GaussianBandit, self).__init__(k)
        self._rng = rng 
        self.mu = mu 
        self.sigma = sigma 
        self.reset() 
        
    def reset(self): 
        self.locs = self._rng.normal(loc=self.mu, scale=self.sigma, size=self.k) 
        self.scales = self._rng.exponential(scale=self.sigma, size=self.k)
        self.optimal = np.argmax(self.locs)
        self.mean_optimal_reward = np.max(self.locs)
        
    def pull(self, action): 
        return (self._rng.normal(loc=self.locs[action], scale=0.1),
                action == self.optimal)
        
        