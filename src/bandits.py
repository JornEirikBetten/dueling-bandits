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
        
    def calculate_true_preference(self): 
        self.P = np.zeros((self.k, self.k))
        
        
class BernoulliBandit(MultiArmedBandit): 
    def __init__(self, k, rng):
        super(BernoulliBandit, self).__init__(k)
        self._rng = rng 
        
    def reset(self): 
        self.action_values = self._rng.random(size=(self.k))
        self.optimal = np.argmax(self.action_values) 
        self.mean_optimal_reward = np.max(self.action_values)
        
    def pull(self, action):
        draw = self._rng.random() 
        return (draw < self.action_values[action], 
                action == self.optimal) 
        
        
class BTBandit: 
    def __init__(self, k, rng, mu=10, sigma=1): 
        self.k = k 
        self.condorcet_winner = 0 
        self._rng = rng 
        self.mu = mu 
        self.sigma = sigma 
        self.reset() 
        
    def reset(self): 
        self.values = self._rng.normal(loc=self.mu, scale=self.sigma, size=(self.k))
        self.condorcet_winner = np.argmax(self.values)
        self.P = self.calculate_true_preference()
        condorcet_row = self.P[self.condorcet_winner, :]
        condorcet_sum = np.sum(condorcet_row > 0.5001)/(self.k - 1)
        print("Condorcet: ", self.condorcet_winner)
        self.regrets = np.zeros((self.k, self.k)) 
        for action in range(self.k):
            row = self.P[action, :]
            row_sum = np.sum(row > 0.5001)/(self.k - 1)
            for second_action in range(self.k): 
                second_row = self.P[second_action, :]
                second_row_sum = np.sum(second_row > 0.5001)/(self.k - 1)
                self.regrets[action, second_action] = 2*condorcet_sum - row_sum - second_row_sum
        #print(self.regrets)
        #print(self.regrets[self.condorcet_winner])
        
    def compare(self, first, second): 
        return (self._rng.random() < self.P[first, second], first == self.condorcet_winner, self.regrets[first, second])
        
    def calculate_true_preference(self): 
        P = np.zeros((self.k, self.k))
        # Bradley-Terry model for preference 
        for i in range(self.k): 
            for j in range(self.k): 
                P[i, j] = self.values[i]/(self.values[i] + self.values[j])
        return P    
        