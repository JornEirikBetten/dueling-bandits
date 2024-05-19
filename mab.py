import matplotlib.pyplot as plt 
import src 
import numpy as np 
import plotnine as p9 
import pandas as pd 
import seaborn as sns
#print(plt.style.available)
plt.style.use('ggplot')
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


loc = 10
scale = 10
nrounds = 10000
rng = np.random.default_rng(12)

bandit = src.GaussianBandit(10, rng, mu=loc, sigma=scale)

agents = [src.Agent(bandit, src.Random(rng), prior=0), 
          src.Agent(bandit, src.EpsilonGreedy(0.1, rng), prior=0), 
          src.Agent(bandit, src.EpsilonGreedy(0.5, rng), prior=0), 
          src.Agent(bandit, src.Adaptive(1, rng, nrounds), prior=0)]

env = src.Environment(bandit, agents, label="Gaussian Bandit")

scores, is_optimal = env.run(nrounds)

rewards_ucb, mean_rewards, active_actions = env.successive_elimination(nrounds)



df_data = {}
names = ["random", "eps_0.1", "eps_0.5", "adaptive"]
labels = ["random", "$\epsilon=0.1$", "$\epsilon=0.5$", "adaptive"]
for i, name in enumerate(names): 
    rewards = scores[:, i]
    optimals = is_optimal[:, i]
    moving_mean_rewards = [np.mean(rewards[:i]) for i in range(1, len(rewards), 1)]
    mean_optimals = [np.mean(optimals[:i]) for i in range(1, len(optimals), 1)]
    #df_data["rewards_" + name] = rewards 
    df_data["mean_rewards_" + name] = moving_mean_rewards
    #df_data["optimals_" + name] =  optimals 
    df_data["mean_optimals_" + name] = mean_optimals
df_data["index"] = [i for i in range(1, len(rewards), 1)]
df = pd.DataFrame(data=df_data)

fig = plt.figure()
ax = plt.gca() 
ax.spines.bottom.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.grid(True)
#ax.set_facecolor('xkcd:salmon')
for i, name in enumerate(names): 
    plt.plot(df_data["index"], df["mean_rewards_" + name], label=labels[i], alpha=0.9)
plt.plot(df_data["index"], [np.mean(rewards_ucb[:i]) for i in range(1, len(rewards), 1)], label="UCB", alpha=0.5)
plt.legend()
ax.set_xlabel("rounds")
ax.set_ylabel("mean accumulated rewards")
plt.savefig("rewards.pdf", format="pdf", bbox_inches="tight")


fig = plt.figure()
ax = plt.gca() 
ax.spines.bottom.set_visible(False)
ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)
ax.spines.left.set_visible(False)
ax.grid(True)
#ax.set_facecolor('xkcd:salmon')
for i, name in enumerate(names): 
    plt.plot(df_data["index"], df["mean_optimals_" + name], label=labels[i])
plt.legend()
ax.set_xlabel("rounds")
ax.set_ylabel("optimal fraction")
plt.savefig("optimals.pdf", format="pdf", bbox_inches="tight")




#Actor-critic 
def one_step_rollout(state):
    Q_vals = []
    for action in actions: 
        new_state = env.step(action)
        logits, critic = forward(new_state)
        Q_vals.append(critic)
