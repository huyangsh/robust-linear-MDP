import numpy as np
from math import sin, cos, pi
from . import RMDP

# Define reward_src vectors.
def get_reward_src(env_name):
    if env_name == "Toy-4":
        return np.array([-10,2,1,2])
    elif env_name == "Toy-10":
        return np.array([0,0,-1,2,-1,0,0,-10,5,-10,0,0.9,1,0])
    else:
        raise NotImplementedError

# Toy environment build functions.
def build_toy_env(reward_src, p_perturb, gamma, thres=1e-5):
    num_states  = reward_src.shape[0]
    num_actions = 3    # 0 = left, 1 = stay, 2 = right.
    
    reward = np.zeros(shape=(num_states,num_actions), dtype=np.float64)
    prob = np.zeros(shape=(num_states,num_actions,num_states), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            reward[s,a] = reward_src[s]
            
            prob[s,a,(s+a-1)%num_states] = 1 - 2*p_perturb
            prob[s,a,(s+a-2)%num_states] = p_perturb
            prob[s,a,(s+a)%num_states]   = p_perturb

    distr_init = np.ones(shape=(num_states,), dtype=np.float32) / num_states

    return RMDP(num_states, num_actions, distr_init, reward, prob, gamma, thres)


# Linear environment build functions.
def get_linear_param(env_name):
    if env_name == "Toy-4":
        distr_init = np.ones(shape=(4,), dtype=np.float32) / 4
        
        return np.array([-10,2,1,2])
    elif env_name == "Mixture":
        pass
    else:
        raise NotImplementedError