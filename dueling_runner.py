import matplotlib.pyplot as plt 
import src 
import numpy as np 
import pandas as pd 
import seaborn as sns
#print(plt.style.available)
plt.style.use('ggplot')


K = 5
M = 100 
loc = 0
scale = 1
nrounds = 10000
rng = np.random.default_rng(12123)

bandit = src.BTBandit(K, rng, mu=loc, sigma=scale)

env = src.DuelingBanditSolver(bandit, rng)

names = ["interleaved-filter", "double-thompson"]

colors = ["tab:red", "tab:cyan", "tab:green", "tab:brown", "tab:orange", "tab:blue", "tab:olive", "tab:pink", "tab:magenta"]
optimal_means = np.zeros(M) 
optimal_actions = np.zeros(M)
#all_rewards = []
#all_optimals = np.zeros((len(names), M, nrounds))
#for m in range(M): 
env.reset() 
#print(env.bandit.P)
#optimal_means[m] = env.bandit.mean_optimal_reward
#optimal_actions[m] = env.bandit.optimal
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5)) 
axs[0].set_xlabel(r"comparisons, $c$")
axs[0].set_ylabel(r"regret-per-comparison, $r(c)/c$")
axs[1].set_xlabel(r"comparisons, $c$")
axs[1].set_ylabel(r"fraction first chosen equals condercet winner")
for algo, name in enumerate(names): 
    
    if name == "interleaved-filter": 
        best_action, regrets, condorcets, comps = env.interleaved_filter(nrounds)
    elif name == "double-thompson": 
        best_action, regrets, condorcets, comps = env.double_thompson(nrounds)
    #print(len(regrets))
    #print(comps)
    axs[0].plot([i for i in range(comps)], [np.mean(regrets[:i+1]) for i in range(comps)], color=colors[algo], label=name)
    axs[1].plot([i for i in range(comps)], [np.mean(condorcets[:i+1]) for i in range(comps)], color=colors[algo], label=name)

xlabs = [f"{i*comps/5:.0f}" for i in range(6)]

axs[0].set_xticks([i*comps/5 for i in range(6)], xlabs)
axs[1].set_xticks([i*comps/5 for i in range(6)], xlabs)
axs[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
axs[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.legend() 
plt.savefig("mean_regret.pdf", format="pdf", bbox_inches="tight")


