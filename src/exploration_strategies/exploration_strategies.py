import numpy as np 




def explore_first(actions, N, T, simulate): 
    all_rewards = []
    rewards = np.zeros((len(actions), N)) 
    # Try each arm N times 
    for a, action in enumerate(actions): 
        for n in range(N): 
            reward = simulate(action)    
            rewards[a, n] = reward     
            all_rewards.append(reward)
    expected_rewards = np.mean(rewards, axis=-1)
    # Select arm with highest rewards 
    best_action = actions[np.argmax(expected_rewards)]
    # Play arm in all remaining rounds
    for t in range(T-N*len(actions)): 
        reward = simulate(action)
        all_rewards.append(reward)
        
    return best_action, all_rewards
    
    
def epsilon_greedy(actions, T, simulate, rng): 
    all_rewards = [] 
    rewards = {}
    for action in actions: 
        rewards[f"action_{action}"] = []
    
    epsilons = [1/t**(1./3) for t in range(T)]    
    action = rng.choice(actions)
    reward = simulate(action)
    all_rewards.append(reward)
    rewards[f"action_{action}"].append(reward)
    for t in range(1, T, 1): 
        r = rng.random() 
        if r > epsilons[t]: 
            action = rng.choice(actions)
        else: 
            action = 