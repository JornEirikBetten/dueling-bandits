import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
            
            
class BanditSolver: 
    """
            MULTI-ARMED BANDIT SOLVER 
    """
    def __init__(self, bandit, rng): 
        self.bandit = bandit 
        self.rng = rng 
        
    def reset(self):
        self.bandit.reset()
    
    def explore_first(self, nrounds): 
        k = self.bandit.k 
        nexplores = int((nrounds/k)**(2./3))
        mean_rewards = np.zeros(k)
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        n_trials = np.zeros(k)
        t = 0 
        # Explore phase 
        while t < nexplores*k: 
            for action in range(k): 
                reward, is_optimal = self.bandit.pull(action)
                n_trials[action] += 1 
                rewards[t] = reward 
                optimals[t] = is_optimal 
                mean_rewards[action] = mean_rewards[action]*(n_trials[action]-1)/n_trials[action] + reward/n_trials[action]
                t += 1 

        # Exploit phase 
        best_action = np.argmax(mean_rewards)
        while t < nrounds: 
            reward, is_optimal = self.bandit.pull(best_action)
            n_trials[best_action] += 1 
            rewards[t] += reward 
            optimals[t] += 1 
            mean_rewards[best_action] = mean_rewards[best_action]*(n_trials[best_action]-1)/n_trials[best_action] + reward/n_trials[best_action]
            t += 1 
            
        return best_action, rewards, optimals 
    
    def epsilon_greedy(self, nrounds, epsilon): 
        k = self.bandit.k 
        mean_rewards = np.zeros(k)
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        n_trials = np.zeros(k)
        t = 0 
        # Sweep all actions once 
        for action in range(k): 
            reward, is_optimal = self.bandit.pull(action)
            rewards[t] = reward
            optimals[t] = is_optimal
            mean_rewards[action] = reward 
            n_trials[action] = 1  
            t += 1 
            
        # Epsilon-greedy 
        while t < nrounds: 
            random_number = self.rng.random() 
            if random_number < epsilon: 
                action = self.rng.choice(k)
            else: 
                action = np.argmax(mean_rewards)
            reward, is_optimal = self.bandit.pull(action)
            rewards[t] = reward 
            optimals[t] = is_optimal
            mean_rewards[action] = mean_rewards[action]*(n_trials[action] - 1)/n_trials[action] + reward/n_trials[action]
            n_trials[action] += 1 
            t += 1
        best_action = np.argmax(mean_rewards)    
        return best_action, rewards, optimals 
    
    def epsilon_greedy_adaptive(self, nrounds): 
        ts = np.array([t for t in range(nrounds)])
        # Set up epsilons
        epsilons = np.where(ts>0, 1/(ts**(1./3)), 1)
        
        k = self.bandit.k 
        mean_rewards = np.zeros(k)
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        n_trials = np.zeros(k)
        
        t = 0 
        # Sweep all actions once 
        for action in range(k): 
            reward, is_optimal = self.bandit.pull(action)
            rewards[t] = reward
            optimals[t] = is_optimal
            mean_rewards[action] = reward 
            n_trials[action] = 1  
            t += 1 
        
        # Epsilon-greedy 
        while t < nrounds: 
            random_number = self.rng.random() 
            if random_number < epsilons[t]: 
                action = self.rng.choice(k)
            else: 
                action = np.argmax(mean_rewards)
            reward, is_optimal = self.bandit.pull(action)
            rewards[t] = reward 
            optimals[t] = is_optimal
            mean_rewards[action] = mean_rewards[action]*(n_trials[action] - 1)/n_trials[action] + reward/n_trials[action]
            n_trials[action] += 1 
            t += 1
        best_action = np.argmax(mean_rewards) 
        return best_action, rewards, optimals 
    
    def upper_confidence_bound(self, nrounds): 
        k = self.bandit.k 
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        n_trials = np.zeros(k)
        mean_rewards = np.zeros(k)
        total_rewards = np.zeros(k)
        uncertainties = np.zeros(k)
        ucbs = np.zeros(k)
        # Sweep all actions once
        for action in range(k): 
            reward, is_optimal = self.bandit.pull(action)
            rewards[action] = reward 
            optimals[action] = is_optimal
            n_trials[action] += 1 
            total_rewards[action] += reward 
            mean_rewards[action] = reward 
            uncertainties[action] = np.sqrt(2*np.log(nrounds))
            ucbs[action] = mean_rewards[action] + uncertainties[action]
        # Pick maximum UCB
        for t in range(k, nrounds, 1): 
            action = np.argmax(ucbs)
            reward, is_optimal = self.bandit.pull(action)
            rewards[t] = reward; optimals[t] = is_optimal
            n_trials[action] += 1 
            total_rewards[action] += reward
            mean_rewards[action] = total_rewards[action]/n_trials[action]
            uncertainties[action] = np.sqrt(2*np.log(nrounds)/n_trials[action])
            ucbs[action] = mean_rewards[action] + uncertainties[action]
        best_action = np.argmax(ucbs)
        return best_action, rewards, optimals
        
            
    def successive_elimination(self, nrounds):
        k = self.bandit.k 
        mean_rewards = np.zeros(self.bandit.k) 
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        n_trials = np.zeros(self.bandit.k)
        total_rewards = np.zeros(k)
        upper_confidence_bounds = np.zeros(self.bandit.k)
        lower_confidence_bounds = np.zeros(self.bandit.k)
        active_actions = [action for action in range(self.bandit.k)]
        t = 0 
        while t < nrounds: 
            # Gathering/exploitation phase 
            for action in active_actions: 
                reward, is_optimal = self.bandit.pull(action)
                rewards[t] = reward
                optimals[t] = is_optimal
                t += 1 
                if t == nrounds: 
                    break
                n_trials[action] += 1
                total_rewards[action] += reward 
                mean_rewards[action] = total_rewards[action]/n_trials[action]
                r = np.sqrt(2*np.log(nrounds)/n_trials[action])
                upper_confidence_bounds[action] = mean_rewards[action] + r 
                lower_confidence_bounds[action] = mean_rewards[action] - r 
            # Elimination phase
            maximum_lower_bound = np.max(lower_confidence_bounds)
            for i, action in enumerate(active_actions): 
                if upper_confidence_bounds[action] < maximum_lower_bound: 
                    action = active_actions.pop(i)
        best_action = active_actions[0]
        return best_action, rewards, optimals
    
    def bernoulli_bandits(self, nrounds): 
        """Bernoulli-bandit algorithms: 
        r = {0, 1} \sim {p_action}

        Args:
            nrounds (_type_): _description_

        Returns:
            _type_: _description_
        """
        k = self.bandit.k 
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        n_trials = np.zeros(self.bandit.k)
        # Define priors 
        alphas = np.ones(k)
        betas = np.ones(k)
        for t in range(nrounds): 
            samples = np.zeros(k)
            for action in range(k): 
                samples[action] = self.rng.beta(alphas[action], betas[action])
            action = np.argmax(samples)
            reward, is_optimal = self.bandit.pull(action)
            rewards[t] = reward 
            optimals[t] = is_optimal
            alphas[action] += reward 
            betas[action] += 1 - reward 
        samples = np.zeros(k)
        for action in range(k): 
            samples[action] = self.rng.beta(alphas[action], betas[action])
        best_action = np.argmax(samples)
        return best_action, rewards, optimals
                
                
    def gaussian_reward_model(self, nrounds): 
        """Gaussian reward model. For each action the reward is modeled
        with a mean and standard deviation, given by the maximum likelihood 
        Gaussian given rewards. 

        Args:
            nrounds (_type_): _description_
        """
        k = self.bandit.k 
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        means = np.zeros(k)
        sigmas = np.zeros(k)
        # Update moments
        n_trials = np.zeros(k)
        total_rewards = np.zeros(k)
        total_rewards_squared = np.zeros(k)
        # One sweep for priors
        for action in range(k): 
            reward, is_optimal = self.bandit.pull(action)
            rewards[action] = reward 
            optimals[action] = is_optimal
            means[action] = reward 
            # Update moments
            n_trials[action] += 1 
            total_rewards[action] = reward 
            total_rewards_squared[action] = reward*reward 

        for t in range(k, nrounds, 1): 
            samples = np.zeros(k)
            for action in range(k): 
                samples[action] = self.rng.normal(loc=means[action], scale=sigmas[action])
            action = np.argmax(samples)    
            reward, is_optimal = self.bandit.pull(action) 
            rewards[t] = reward 
            optimals[t] = is_optimal
            # Update moments
            n_trials[action] += 1
            total_rewards[action] += reward 
            total_rewards_squared[action] += reward*reward 
            means[action] = total_rewards[action]/n_trials[action]
            sigmas[action] = np.sqrt(n_trials[action]*total_rewards_squared[action] - total_rewards[action]*total_rewards[action])/n_trials[action]   
        
        best_action = action 
        return best_action, rewards, optimals
    
    def boltzmann_reward_model(self, nrounds): 
        """Boltzmann reward model. For each action, the mean reward is collected and 
        the action distribution is modeled as a Boltzmann
        Args:
            nrounds (_type_): _description_
        """
        k = self.bandit.k 
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        means = np.zeros(k)
        sigmas = np.zeros(k)
        # Update moments
        n_trials = np.zeros(k)
        total_rewards = np.zeros(k)
        # One sweep for priors
        for action in range(k): 
            reward, is_optimal = self.bandit.pull(action)
            rewards[action] = reward 
            optimals[action] = is_optimal
            means[action] = reward 
            # Update moments
            n_trials[action] += 1 
            total_rewards[action] = reward 
            

        for t in range(k, nrounds, 1): 
            probabilities = np.exp(means*10)/np.sum(np.exp(means*10))
            action = self.rng.choice(k, p=probabilities)    
            reward, is_optimal = self.bandit.pull(action) 
            rewards[t] = reward 
            optimals[t] = is_optimal
            # Update moments
            n_trials[action] += 1
            total_rewards[action] += reward 
            means[action] = total_rewards[action]/n_trials[action]
        
        best_action = action 
        return best_action, rewards, optimals
            
# TODO: Interleaved filter, double Thompson sampling. 
        
class DuelingBanditSolver: 
    """
            DUELING BANDITS SOLVER 
    Algorithms:   
        Interleaved filter 
        Double Thompson Sampling       
    """
    def __init__(self, bandit, rng): 
        self.bandit = bandit 
        self.rng = rng 
        
        
        
    def reset(self):
        self.bandit.reset()
        
    def interleaved_filter(self, nrounds):
        """Dueling bandits interleaved filter algorithm. (The K -armed dueling bandits problem, Journal of Computer and System Sciences 78 (2012) 1538–1556)

        Args:
            nrounds (int): number of rounds run. 
        """
        k = self.bandit.k  
        delta_inv = nrounds*k*k 
        active_actions = [action for action in range(k)]
        top_action = active_actions.pop(self.rng.choice(k))
        ncomparisons = 0 
        ncomparisons_top = np.zeros((k))
        wins = np.zeros((k))
        win_fractions = np.zeros((k))
        win_confidences = np.zeros((k))
        regrets = [] 
        condorcets = [] 
        t = 0 
        while t < nrounds:
            if active_actions == []: 
                # If empty, compare with itself
                win, condorcet, regret = self.bandit.compare(top_action, top_action)
                wins[action] += win   
                ncomparisons += 1
                ncomparisons_top[action] += 1
                win_fractions[action] = wins[action]/ncomparisons_top[action]
                win_confidences[action] = np.sqrt(4*np.log(delta_inv)/ncomparisons_top[action])
                condorcets.append(condorcet); regrets.append(regret)
            else: 
                for action in active_actions: 
                    win, condorcet, regret = self.bandit.compare(top_action, action)
                    wins[action] += win
                    ncomparisons += 1
                    ncomparisons_top[action] += 1
                    win_fractions[action] = wins[action]/ncomparisons_top[action]
                    win_confidences[action] = np.sqrt(4*np.log(delta_inv)/ncomparisons_top[action])
                    condorcets.append(condorcet); regrets.append(regret)
            t += 1
                
            # Declare top action winner against action  
            for idx, action in enumerate(active_actions): 
                larger_than_half = win_fractions[action] > 0.5 
                half_below_confidence = win_fractions[action] - win_confidences[action] > 0.5 
                # Prune actions that are confidently worse than top action
                if larger_than_half and half_below_confidence:
                    #print(f"Remove action={action}")
                    #print(f"P=({win_fractions[action] - win_confidences[action]}, {win_fractions[action]+win_confidences[action]})") 
                    active_actions.pop(idx)
            
            
            for idx, action in enumerate(active_actions):
                half_above_confidence = win_fractions[action] + win_confidences[action] < 0.5 
                # Exchange top action to action that is confidently better than current best
                if not larger_than_half and half_above_confidence: 
                    #print(action)
                    #print(f"P=({win_fractions[action] - win_confidences[action]}, {win_fractions[action]+win_confidences[action]})")
                    top_action = action
                    active_actions.pop(idx) 
                    wins = np.zeros((k)) 
                    ncomparisons_top = np.zeros(k) 
                    win_fractions = np.zeros(k)
                    win_confidences = np.zeros(k)
                    break
        best_action = top_action
        return best_action, regrets, condorcets, ncomparisons
                                
                                
    # def copeland_confidence_bound(self, nrounds, alpha):
    #     """CCB algorithm. http://arxiv.org/abs/1506.00312
        
    #     """ 
    #     k = self.bandit.k 
    #     wins = np.zeros((k, k))
    #     for i in range(k): 
    #         wins[i, i] = 0.5 
    #     potential_winners = [a for a in range(k)]
    #     potential_to_beat_winner = []
    #     # Estimated max losses of a Copeland winner 
    #     l_C = k 
    #     for t in range(nrounds): 
    #         upper_bound = (wins + alpha*np.log(t+1))/(wins + wins.T)
    #         lower_bound = (wins - alpha*np.log(t+1))/(wins + wins.T)
    #         for i in range(k): 
    #             upper_bound[i, i] = 0.5 
    #             lower_bound[i, i] = 0.5 
            
            
    #         upper_cw = [np.sum(np.where(upper_bound[contestant, :] >= 0.5)) - 1 for contestant in range(k)]
    #         lower_cw = [np.sum(np.where(lower_bound[contestant, :] >= 0.5)) - 1 for contestant in range(k)]
    #         cw = np.argmax(upper_cw)
    #         # Reset disproven hypothesis 
            
    #         # Remove non-Copeland winners 
            
    #         # Add Copeland winners 
    
    
    
    def double_thompson(self, nrounds): 
        k = self.bandit.k 
        W = np.zeros((k, k))
        alpha = 2
        t = 0 
        condorcets = []; regrets = [] 
        while t < nrounds: 
            #print(W)
            if t == 0: 
                U = np.ones((k, k)); L = np.zeros((k, k))
                for a in range(k): 
                    U[a, a] = 0.5; L[a, a] = 0.5 
            else: 
                U = (W + alpha*np.log(t+1)); L = (W - alpha*np.log(t+1))
                for i in range(k): 
                    for j in range(k): 
                        if (W + W.T)[i, j] == 0: 
                            U[i, j] = 1 
                            L[i, j] = 0 
                        else: 
                            U[i, j] = U[i, j]/(W + W.T)[i, j]
                            L[i, j] = L[i, j]/(W + W.T)[i, j]
                    U[i, i] = 0.5; L[i, i] = 0.5
            upper_copelands = []
            for a in range(k): 
                    row = U[a, :]
                    row_sum = np.sum(row > 0.5001)
                    upper_copelands.append(row_sum/(k - 1))    
            C = np.argmax(upper_copelands)
            thetas = np.zeros((k, k))
            thetasums = []        
            thetaidxs = [] 
            for i in range(k):
                thetarow = thetas[i, :]
                if i  ==  C: 
                    thetasums.append(np.sum(thetarow>0.5001)) 
                    thetaidxs.append(i)
                
                for j in range(i+1): 
                    thetas[i, j] = self.rng.beta(W[i, j]+1, W[j, i] + 1)
                    thetas[j, i] = 1 - thetas[i, j]
            
            idx = np.argmax(thetasums)
            first_action = thetaidxs[idx]
            thetas = np.zeros(k)
            for j in range(k): 
                if j == first_action: 
                    thetas[first_action] = 0.5 
                else: 
                    thetas[j] = self.rng.beta(W[j, first_action] + 1, W[first_action, j] + 1)
                    
            available_actions = [j for j in range(k)]
            available_actions.pop(first_action)
            uncertain_pairs = [j for j in available_actions if L[j, first_action]<=0.5]
            uncertain_thetas = [thetas[j] for j in uncertain_pairs]
            idx = np.argmax(uncertain_thetas)
            second_action = uncertain_pairs[idx]
            win, condorcet, regret = self.bandit.compare(first_action, second_action)
            if win: 
                W[first_action, second_action] += 1 
            else: 
                W[second_action, first_action] += 1 
            condorcets.append(condorcet); regrets.append(regret)
            t += 1 
        best_action = first_action
        return best_action, regrets, condorcets, nrounds