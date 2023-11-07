import numpy as np
import random

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
eps = 0

env_name = "Toy-4"
reward_src = get_reward_src(env_name)
env = build_toy_env(reward_src, 0, gamma, THRES)

pi = np.array(
[[0.5       ,0.        ,0.5       ],
 [0.        ,0.46897326,0.53102674],
 [0.33581501,0.32837   ,0.33581499],
 [0.53102675,0.46897325,0.        ]]
)
Q_rob = env.robust_Q(pi=pi, eps=0)
Q_dp = env.DP_Q(pi=pi)
print(print_float_matrix(Q_rob*(pi>0)))
print(print_float_matrix(Q_dp*(pi>0)))
print(np.linalg.norm((Q_rob-Q_dp)*pi).flatten())