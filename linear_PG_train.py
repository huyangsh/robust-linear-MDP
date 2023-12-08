import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from agent import LinearPolicyGradientAgent
from env import LinearMDP, get_linear_param
from utils import print_float_list, print_float_matrix


THRES = 1e-5
T_EST = 100
T_Q   = 100

seed = 0
random.seed(seed)
np.random.seed(seed)
# torch.manual_seed(seed)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

GAMMA     = 0.95
EPS_PHI   = 0
EPS_THETA = 0
EPS_MU    = 0
USE_DUAL  = False

eta = 0.02
T   = int(5e3)
print(f"learning rate: {eta}.")


# Build environment
env_name = "Toy-4"
distr_init, phi, theta, mu, mu_perturb = get_linear_param(env_name)
env = LinearMDP(distr_init, phi, theta, mu, GAMMA, eps_phi=EPS_PHI, eps_theta=EPS_THETA, eps_mu=EPS_MU, thres=1e-5)
test_env = LinearMDP(distr_init, phi, theta, mu_perturb, GAMMA, thres=1e-5)

# Reference policies.
pi_nominal = env.get_opt_pi()
pi_perturb = test_env.get_opt_pi()
reward_nominal = test_env.distr_init @ test_env.DP_pi(pi=pi_nominal)
reward_perturb = test_env.distr_init @ test_env.DP_pi(pi=pi_perturb)

tqdm.write(f"pi_nominal:\n{print_float_matrix(pi_nominal.T)}\nreward = {reward_nominal}.\n")
tqdm.write(f"pi_perturb:\n{print_float_matrix(pi_perturb.T)}\nreward = {reward_perturb}.")

agent = LinearPolicyGradientAgent(env, eta, T_EST, T_Q, THRES, use_dual=USE_DUAL)

pi_init = np.ones(shape=(env.num_states, env.num_actions), dtype=np.float32) / env.num_actions
agent.reset(pi_init)

eval_freq = 10
try:
    reward_list, pi_list, t_list = [], [], []
    bar = tqdm(range(T))
    bar.set_description_str(f"eta = {eta}, eps = ({EPS_PHI}, {EPS_THETA}, {EPS_MU}), target_reward = [{reward_nominal:.4g}, {reward_perturb:.4g}]")
    for t in bar:
        pi, info = agent.update()
        pi_list.append(pi)

        if t % eval_freq == 0:
            tqdm.write(f"Evaluate at iteration #{t}:")
            tqdm.write(f"Q_pi:\n{print_float_matrix(info['Q_pi'].T)}")
            tqdm.write(f"pi:\n{print_float_matrix(pi.T)}")
            tqdm.write(f"V_pi:\n{print_float_list(env.Q_to_V(info['Q_pi'],pi))}")
            t_list.append(t)

            test_reps = 10
            test_T = 1000
            cur_rewards = []
            for rep in range(test_reps):
                cur_rewards.append( test_env.eval(agent.select_action, T_eval=test_T) )
            tqdm.write(f"\ntest rewards:\n{print_float_list(cur_rewards)}\n")

            V_pi = test_env.DP_pi(pi=pi)
            avg_reward = test_env.distr_init @ V_pi
            tqdm.write(f"average reward: {avg_reward}\n\n")
            reward_list.append(avg_reward)
except KeyboardInterrupt:
    pass

# Plot rewards.
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.plot(t_list, reward_list, label=r"$\varepsilon=$" + f"{EPS_PHI, EPS_THETA, EPS_MU}")
ax.axhline(y=reward_nominal, linestyle="--", color="r", label="nominal optimal")
ax.axhline(y=reward_perturb, linestyle="--", color="g", label="perturbed optimal")
ax.legend(loc="lower right")
fig.savefig(f"./log/linear_PG_{env_name}_{timestamp}_reward_{EPS_PHI:.3f}_{EPS_THETA:.3f}_{EPS_MU:.3f}_{GAMMA:.2f}_{eta:.2f}.png", dpi=300)

# Plot policy changes over time.
if env_name == "Toy-4":
    fig = plt.figure(figsize=(20,5))
    t_horizon = list(range(len(pi_list)))
    for i in range(4):
        ax = fig.add_subplot(1,4,i+1)
        ax.plot(t_horizon, [pi[i,0] for pi in pi_list], label="left")
        ax.plot(t_horizon, [pi[i,1] for pi in pi_list], label="stay")
        ax.plot(t_horizon, [pi[i,2] for pi in pi_list], label="right")
        ax.legend()
        ax.set_xlabel(f"State {i}")
        ax.set_ylabel(f"probability")
    fig.savefig(f"./log/linear_PG_{env_name}_{timestamp}_policy_{EPS_PHI:.3f}_{EPS_THETA:.3f}_{EPS_MU:.3f}_{GAMMA:.2f}_{eta:.2f}.png", dpi=300)
elif env_name == "Mixture":
    fig = plt.figure(figsize=(20,5))
    t_horizon = list(range(len(pi_list)))
    for i in range(3):
        ax = fig.add_subplot(1,3,i+1)
        ax.plot(t_horizon, [pi[i,0] for pi in pi_list], label="high risk")
        ax.plot(t_horizon, [pi[i,1] for pi in pi_list], label="low risk")
        ax.legend()
        ax.set_xlabel(f"State {i}")
        ax.set_ylabel(f"probability")
    fig.savefig(f"./log/linear_PG_{env_name}_{timestamp}_policy_{EPS_PHI:.3f}_{EPS_THETA:.3f}_{EPS_MU:.3f}_{GAMMA:.2f}_{eta:.2f}.png", dpi=300)
else:
    print("Policy plotting error: not implemented.")