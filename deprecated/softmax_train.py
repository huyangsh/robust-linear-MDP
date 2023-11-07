import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from agent import SoftmaxPolicyGradientAgent
from env import TabularMDP, get_reward_src, build_toy_env
from utils import print_float_list, print_float_matrix


THRES = 1e-5
T_EST = 100

seed = 0
random.seed(seed)
np.random.seed(seed)
# torch.manual_seed(seed)

# Build environment
p_perturb = 0.05
gamma = 0.95
eps = 0.01

env_name = "Toy-4"
reward_src = get_reward_src(env_name)
env = build_toy_env(reward_src, 0, gamma, THRES)
test_env = build_toy_env(reward_src, p_perturb, gamma, THRES)

eta = 0.05
T   = int(2e3)
print(f"learning rate: {eta}.")

agent = SoftmaxPolicyGradientAgent(env, eps, eta, T_EST, THRES)

pi_init = np.zeros(shape=(env.num_states, env.num_actions), dtype=np.float32)
agent.reset(pi_init)

eval_freq = 20
try:
    reward_list, pi_list, t_list = [], [], []
    bar = tqdm(range(T))
    bar.set_description_str(f"eta = {eta}, eps = {eps}, p = {p_perturb}")
    for t in bar:
        pi, info = agent.update()

        if t % eval_freq == 0:
            tqdm.write(f"Evaluate at iteration #{t}:")
            tqdm.write(f"Q_pi:\n{print_float_matrix(info['Q_pi'].T)}")
            tqdm.write(f"pi:\n{print_float_matrix(pi.T)}")

            t_list.append(t)
            pi_list.append(pi)

            test_reps = 10
            test_T = 1000
            cur_rewards = []
            for rep in range(test_reps):
                cur_rewards.append( test_env.eval(agent.select_action, T_eval=test_T) )
            tqdm.write(f"rewards:\n{print_float_list(cur_rewards)}\n\n")
            reward_list.append(np.array(cur_rewards, dtype=np.float32).mean())
except KeyboardInterrupt:
    pass

fig = plt.figure(figsize=(3*env.num_states,3))
for i in range(env.num_states):
    ax = fig.add_subplot(1,env.num_states,i+1)
    ax.plot(t_list, [pi[i,0] for pi in pi_list], label="left")
    ax.plot(t_list, [pi[i,1] for pi in pi_list], label="stay")
    ax.plot(t_list, [pi[i,2] for pi in pi_list], label="right")
    ax.legend()
    ax.set_xlabel(f"State {i}")
    ax.set_ylabel(f"probability")
fig.savefig(f"softmax_{env_name}_{eps:.2f}_{p_perturb:.2f}_{eta:.2f}_policy.png", dpi=300)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.plot(t_list, reward_list, label="$\varepsilon=0.2$")
fig.savefig(f"softmax_{env_name}_{eps:.2f}_{p_perturb:.2f}_{eta:.2f}_reward.png", dpi=300)