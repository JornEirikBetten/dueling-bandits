import matplotlib.pyplot as plt 
import src 
import numpy as np 
import plotnine as p9 
import pandas as pd 
import seaborn as sns
#print(plt.style.available)
#plt.style.use('ggplot')
from plotnine import (
    ggplot,
    aes,
    geom_line,
    facet_wrap,
    labs,
    scale_x_datetime,
    element_text,
    theme_538
)

K = 5
M = 100 
loc = 0
scale = 1
nrounds = 10000
rng = np.random.default_rng(12123)

#bandit = src.GaussianBandit(K, rng, mu=loc, sigma=scale)
bandit = src.BernoulliBandit(K, rng)

agents = [src.Agent(bandit, src.Random(rng), prior=0), 
          src.Agent(bandit, src.EpsilonGreedy(0.1, rng), prior=0), 
          src.Agent(bandit, src.EpsilonGreedy(0.5, rng), prior=0), 
          src.Agent(bandit, src.Adaptive(1, rng, nrounds), prior=0)]

env = src.BanditSolver(bandit, rng)

#names = ["epsilon-0.1", "epsilon-adaptive", "gaussian-reward", "boltzmann-reward", "thompson-bernoulli"]
names = ["thompson-bernoulli", "explore-first", "$\epsilon=t^{-1/3}$", "successive-elimination", "ucb"]
#names = ["thompson-bernoulli", "explore-first", "ucb"]
colors = ["tab:red", "tab:cyan", "tab:green", "tab:brown", "tab:orange", "tab:blue", "tab:olive", "tab:pink"]
optimal_means = np.zeros(M) 
all_rewards = np.zeros((len(names), M, nrounds)) 
all_optimals = np.zeros((len(names), M, nrounds))
for m in range(M): 
    env.reset() 
    optimal_means[m] = env.bandit.mean_optimal_reward
    for algo, name in enumerate(names): 
        if name == "explore-first":
            nexplores = 10 
            best_action, rewards, optimals = env.explore_first(nrounds)
        elif name == "epsilon-0.1": 
            best_action, rewards, optimals = env.epsilon_greedy(nrounds, 0.1)
        elif name == "epsilon-0.5": 
            best_action, rewards, optimals = env.epsilon_greedy(nrounds, 0.5)
        elif name == "successive-elimination":     
            best_action, rewards, optimals = env.successive_elimination(nrounds)
        elif name == "$\epsilon=t^{-1/3}$": 
            best_action, rewards, optimals = env.epsilon_greedy_adaptive(nrounds)
        elif name == "gaussian-reward": 
            best_action, rewards, optimals = env.gaussian_reward_model(nrounds)
        elif name == "boltzmann-reward": 
            best_action, rewards, optimals = env.boltzmann_reward_model(nrounds)
        elif name == "thompson-bernoulli": 
            best_action, rewards, optimals = env.bernoulli_bandits(nrounds)
        elif name == "ucb": 
            best_action, rewards, optimals = env.upper_confidence_bound(nrounds)
        all_rewards[algo, m, :] = rewards#[np.mean(rewards[:i]) for i in range(nrounds)] 
        all_optimals[algo, m, :] = optimals#[np.mean(optimals[:i]) for i in range(nrounds)]

optimal_means = np.reshape(optimal_means, (M, 1))
fig = plt.figure()
ax = plt.gca() 
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.grid(True)
plt.xlabel("rounds, $t$")
plt.ylabel(r"expected regret-per-round, $\mathbb{E}[r(t)]/T$")
t = [i+1 for i in range(nrounds)]
for algo, name in enumerate(names):
    mean_regret = optimal_means - all_rewards[algo, :, :] 
    expected_regret = [np.mean(np.mean(mean_regret, axis=0)[:i]) for i in range(nrounds)]
    moving_average_exp_regret = [np.mean(expected_regret[i:i+5]) for i in range(len(expected_regret)-5)]
    std_regret = np.std(mean_regret, axis=0) 
    plt.plot(t, expected_regret, label=name, color=colors[algo], alpha=1)
    #plt.plot(t, np.mean(all_rewards[algo, :, :], axis=0), label=name, color=colors[algo], alpha=0.5)
    #plt.plot(t, expected_regret + 1.96*std_regret, linestyle="dashed", color=colors[algo], alpha=0.5)
    #plt.plot(t, expected_regret - 1.96*std_regret, linestyle="dashed", color=colors[algo], alpha=0.5)
    #ax.fill_between(t, expected_regret + 1.96*std_regret, expected_regret - 1.96*std_regret) 
#plt.plot(t[100:], [np.sqrt(K*np.log(tval)/tval) for tval in t[100:]], linestyle="dashed", color=colors[0], label="$\sqrt{K\log{t}}/t$")
#plt.plot(t[100:], [1/(tval**(1./3))*np.sqrt(K*np.log(tval))**(1./3) for tval in t[100:]], linestyle="dashed", color=colors[1], label="$(\sqrt{K\log{t}}/t)^{1/3}$")
#plt.hlines(np.sqrt(K*np.log(nrounds)/nrounds), xmin=1, xmax=t[-1], linestyles="dashed", color=colors[0], label="thompson regret-per-round upper bound")
plt.yticks([0, 0.25, 0.5])
xlabs = [f"{i*len(t)/5:.0f}" for i in range(5)]
xlabs.append("T")
plt.xticks([i*len(t)/5 for i in range(6)], xlabs)
plt.legend() 
plt.savefig("mean_rewards.pdf", format="pdf", bbox_inches="tight")

# for m in range(M): 
#     env.reset() 
#     optimal_means.append(env.bandit.mean_optimal_reward) 
#     data = {}
#     for name in names: 
#         if name == "explore-first":
#             nexplores = 50 
#             best_action, rewards, optimals = env.explore_first(nrounds, nexplores)
#         elif name == "epsilon-0.1": 
#             best_action, rewards, optimals = env.epsilon_greedy(nrounds, 0.1)
#         elif name == "epsilon-0.5": 
#             best_action, rewards, optimals = env.epsilon_greedy(nrounds, 0.5)
#         elif name == "ucb":     
#             best_action, rewards, optimals = env.successive_elimination(nrounds)
#         elif name == "adaptive": 
#             best_action, rewards, optimals = env.epsilon_greedy_adaptive(nrounds)
#         elif name == "gaussian": 
#             best_action, rewards, optimals = env.gaussian_reward_model(nrounds)
#         data[name + f"-rewards-{m}"] = rewards 
#         data[name + f"-optimals-{m}"] = optimals 
#         data[name + f"-mean-rewards-{m}"] = [np.mean(rewards[:i]) for i in range(nrounds)]
#         data[name + f"-mean-optimals-{m}"] = [np.mean(optimals[:i]) for i in range(nrounds)]
#         #print(f"Algorithm {name} found action {best_action} to be the best action.")

#best_action, total_comparisons_made = env.interleaved_filter(1e8)
#print(f"Interleaved filter found action {best_action} to be best using {total_comparisons_made}.")
# df = pd.DataFrame(data=data)
# x = [i+1 for i in range(nrounds)]
# fig = plt.figure() 
# plt.xlabel("rounds")
# plt.ylabel("mean rewards")
# for name in names: 
#     y = df[name + "-mean-rewards"]
#     plt.plot(x, y, label=name)
# plt.legend()
# plt.savefig("mean_rewards.pdf", format="pdf", bbox_inches="tight")    


# fig = plt.figure() 
# plt.xlabel("rounds")
# plt.ylabel("fraction optimal action")
# for name in names: 
#     y = df[name + "-mean-optimals"]
#     plt.plot(x, y, label=name)
# plt.legend() 
# plt.savefig("mean_optimals.pdf", format="pdf", bbox_inches="tight")


# df_data = {}
# names = ["random", "eps_0.1", "eps_0.5", "adaptive"]
# labels = ["random", "$\epsilon=0.1$", "$\epsilon=0.5$", "adaptive"]
# for i, name in enumerate(names): 
#     rewards = scores[:, i]
#     optimals = is_optimal[:, i]
#     moving_mean_rewards = [np.mean(rewards[:i]) for i in range(1, len(rewards), 1)]
#     mean_optimals = [np.mean(optimals[:i]) for i in range(1, len(optimals), 1)]
#     #df_data["rewards_" + name] = rewards 
#     df_data["mean_rewards_" + name] = moving_mean_rewards
#     #df_data["optimals_" + name] =  optimals 
#     df_data["mean_optimals_" + name] = mean_optimals
# df_data["index"] = [i for i in range(1, len(rewards), 1)]
# df = pd.DataFrame(data=df_data)

# fig = plt.figure()
# ax = plt.gca() 
# ax.spines.bottom.set_visible(False)
# ax.spines.top.set_visible(False)
# ax.spines.right.set_visible(False)
# ax.spines.left.set_visible(False)
# ax.grid(True)
# #ax.set_facecolor('xkcd:salmon')
# for i, name in enumerate(names): 
#     plt.plot(df_data["index"], df["mean_rewards_" + name], label=labels[i], alpha=0.9)
# plt.plot(df_data["index"], [np.mean(rewards_ucb[:i]) for i in range(1, len(rewards), 1)], label="UCB", alpha=0.5)
# plt.legend()
# ax.set_xlabel("rounds")
# ax.set_ylabel("mean accumulated rewards")
# plt.savefig("rewards.pdf", format="pdf", bbox_inches="tight")


# fig = plt.figure()
# ax = plt.gca() 
# ax.spines.bottom.set_visible(False)
# ax.spines.top.set_visible(False)
# ax.spines.right.set_visible(False)
# ax.spines.left.set_visible(False)
# ax.grid(True)
# #ax.set_facecolor('xkcd:salmon')
# for i, name in enumerate(names): 
#     plt.plot(df_data["index"], df["mean_optimals_" + name], label=labels[i])
# plt.legend()
# ax.set_xlabel("rounds")
# ax.set_ylabel("optimal fraction")
# plt.savefig("optimals.pdf", format="pdf", bbox_inches="tight")