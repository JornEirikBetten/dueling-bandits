import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 



class Environment: 
    def __init__(self, bandit, agents, label="Multi-Armed Bandit"): 
        self.bandit = bandit 
        self.agents = agents 
        self.label = label 
        
        
        
    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()
        
    def run(self, rounds): 
        scores = np.zeros((rounds, len(self.agents))) 
        optimal = np.zeros_like(scores)
        
        self.reset() 
        for t in range(rounds): 
            for i, agent in enumerate(self.agents): 
                action = agent.pick_action()
                reward, is_optimal = self.bandit.pull(action)
                agent.observe(reward)
                
                scores[t, i] += reward
                optimal[t, i] += is_optimal
                
        return scores, optimal 
    
    def explore_first(self, nrounds, nexplores): 
        k = self.bandit.k 
        if nrounds < nexplores*k:
            print("Not enough rounds to explore that much.") 
            return  0
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
            rewards[t] += 1 
            optimals[t] += 1 
            mean_rewards[best_action] = mean_rewards[best_action]*(n_trials[best_action]-1)/n_trials[best_action] + reward/n_trials[best_action]
            t += 1 
            
        return best_action, rewards, optimals 
            
    def successive_elimination(self, nrounds):
        mean_rewards = np.zeros(self.bandit.k) 
        rewards = np.zeros(nrounds)
        n_trials = np.zeros(self.bandit.k)
        upper_confidence_bounds = np.zeros(self.bandit.k)
        lower_confidence_bounds = np.zeros(self.bandit.k)
        active_actions = [action for action in range(self.bandit.k)]
        t = 0 
        while t < nrounds: 
            for action in active_actions: 
                reward, is_optimal = self.bandit.pull(action)
                rewards[t] = reward
                t += 1 
                n_trials[action] += 1
                mean_rewards[action] = mean_rewards[action]*(n_trials[action]-1)/n_trials[action] + reward/n_trials[action] 
                r = np.sqrt(2*np.log(nrounds)/n_trials[action])
                upper_confidence_bounds[action] = mean_rewards[action] + r 
                lower_confidence_bounds[action] = mean_rewards[action] - r 
            maximum_lower_bound = np.max(lower_confidence_bounds)
            for i, action in enumerate(active_actions): 
                if upper_confidence_bounds[action] < maximum_lower_bound: 
                    action = active_actions.pop(i)
        
        return rewards, mean_rewards, active_actions
                 


# TODO: Thompson sampling for MABs, Double Thompson sampling for dueling. 
            
            
class BanditSolver: 
    def __init__(self, bandit, rng): 
        self.bandit = bandit 
        self.rng = rng 
        
        
        
    def reset(self):
        self.bandit.reset()
    
    
    """
            MULTI-ARMED BANDIT
    """
    
    def explore_first(self, nrounds, nexplores): 
        k = self.bandit.k 
        if nrounds < nexplores*k:
            print("Not enough rounds to explore that much.") 
            return  0
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
            
    def successive_elimination(self, nrounds):
        mean_rewards = np.zeros(self.bandit.k) 
        rewards = np.zeros(nrounds)
        optimals = np.zeros(nrounds)
        n_trials = np.zeros(self.bandit.k)
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
                mean_rewards[action] = mean_rewards[action]*(n_trials[action]-1)/n_trials[action] + reward/n_trials[action] 
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
        alphas = 1000*np.ones(k)
        betas = 1000*np.ones(k)
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
            samples[k] = self.rng.beta(alphas[action], betas[action])
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
            
            #if n_trials[action]>1: 
                #first_term = (n_trials[action]-2)*sigmas[action]**2
                #second_term = (n_trials[action]-1)*(old_mean - means[action])**2
                #final_term = (reward - means[action])**2 
                
                #sigmas[action] = np.sqrt((first_term + second_term + final_term)/(n_trials[action]-1))
            #else: 
                #sigmas[action] = np.sqrt(())
        
        best_action = action 
        return best_action, rewards, optimals
            
    # def thompson_sampling_gaussian(self, nrounds): 
    #     k = self.bandit.k 
    #     means = np.zeros(k)
    #     n_trials = np.zeros(k)
    #     scales = np.zeros(k)
    #     data =  {
    #         "means" = [], 
    #         "scales" = []
    #             }
    #     rewards = np.zeros(nrounds)
    #     optimals = np.zeros(nrounds)
    #     for action in range(k): 
    #         reward, is_optimal = self.bandit.pull(action)
    #         means[action] = reward 
    #         scales[action] = 0 
    #         data.means.append(reward)
    #         data.scales.append(scales[action])
    #         n_trials[action] += 1
    #         optimals[action] = is_optimal 
    #         rewards[k] = reward 
            
            
    #     for t in range(k, nrounds, 1): 
    #         samples = np.zeros(k)
    #         for action in range(k): 
    #              samples[k] = self.rng.normal(loc=means[k], scale=scales[k])
                 
    #         action = np.argmax(samples)
            
    #         n_trials[action] += 1 
    #         reward, is_optimal = self.bandit.pull(action)
    #         rewards[t] = reward 
    #         optimals[t] = is_optimal
    #         means[action] = 
        
        
    
    """
                DUELING BANDITS 
    """
    def interleaved_filter(self, nrounds):
        """Dueling bandits interleaved filter algorithm. 

        Args:
            nrounds (int): number of rounds run. 
        """
        k = self.bandit.k  
        delta_inv = nrounds*k*k 
        active_actions = [action for action in range(k)]
        top_action = active_actions.pop(self.rng.choice(k))
        ncomparisons = 0 
        ncomparisons_top = np.zeros(k)
        wins = np.zeros((k))
        win_fractions = np.zeros((k))
        win_confidences = np.zeros((k))
        t = 0 
        while len(active_actions) > 0 and t < nrounds:
            top_reward, _ = self.bandit.pull(top_action) 
            for action in active_actions: 
                reward, _ = self.bandit.pull(action)
                wins[action] += top_reward>reward     
                ncomparisons += 1
                ncomparisons_top[action] += 1 
                win_fractions[action] = wins[action]/ncomparisons_top[action]
                win_confidences[action] = np.sqrt(4*np.log(delta_inv)/ncomparisons_top[action])
            t += 1
                
            # Declare top action winner against action  
            for idx, action in enumerate(active_actions): 
                larger_than_half = win_fractions[action] > 0.5 
                half_below_confidence = win_fractions[action] - win_confidences[action] > 0.5 
                # Prune actions that are confidently worse than top action
                if larger_than_half and half_below_confidence:
                    print(f"Remove action={action}")
                    print(f"P=({win_fractions[action] - win_confidences[action]}, {win_fractions[action]+win_confidences[action]})") 
                    active_actions.pop(idx)
            
            
            for idx, action in enumerate(active_actions):
                half_above_confidence = win_fractions[action] + win_confidences[action] < 0.5 
                # Exchange top action to action that is confidently better than current best
                if not larger_than_half and half_above_confidence: 
                    print(action)
                    print(f"P=({win_fractions[action] - win_confidences[action]}, {win_fractions[action]+win_confidences[action]})")
                    top_action = action
                    active_actions.pop(idx) 
                    wins = np.zeros(k)
                    ncomparisons_top = np.zeros(k) 
                    win_fractions = np.zeros(k)
                    win_confidences = np.zeros(k)
                    break
            #for action in active_actions: 
            #    print(f"P=({win_fractions[action] - win_confidences[action]}, {win_fractions[action]+win_confidences[action]})")
        best_action = top_action
        return best_action, ncomparisons
                                
            
    