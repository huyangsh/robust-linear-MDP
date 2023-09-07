import numpy as np
import scipy.optimize.minimize as minimize
import random
from copy import deepcopy
from math import exp, log

from . import Env


class RMDP(Env):
    def __init__(self, num_states, num_actions, distr_init, reward, prob, gamma, thres=1e-5):
        assert distr_init.shape == (num_states,)
        assert reward.shape == (num_states, num_actions)
        assert prob.shape == (num_states, num_actions, num_states)
        assert gamma <= 1

        self.num_states  = num_states
        self.num_actions = num_actions
        self.states      = np.arange(self.num_states)
        self.actions     = np.arange(self.num_actions)
        self.distr_init  = distr_init

        self.dim_state   = 1
        self.dim_action  = 1

        self.reward = reward
        self.prob   = prob

        self.gamma  = gamma
        self.thres  = thres


    # Environment functions (compatible with OpenAI gym).
    def reset(self):
        self.state = random.choices(self.states, weights=self.distr_init)[0]
        return np.array([self.state], dtype=np.float32)
    
    def step(self, action):
        reward = self.reward[self.state, action]
        self.state = random.choices(self.states, weights=self.prob[self.state,action,:])[0]
        return np.array([self.state], dtype=np.float32), reward, False, None    # Compatible with the OpenAI gym interface: done = False (non-episodic).


    # Utility: robust policy evaluation.
    def _index(self, s, a):
        return (s-1)*self.num_actions + a
    
    def _index_xi(self, s, a):
        return (self.num_states+s-1)*self.num_actions + a

    def robust_Q(self, pi):
        def func(x):
            J = 0
            for s in self.states:
                for a in self.actions:
                    J += self.distr_init[s] * pi[s,a] * x[self._index(s,a)]
            return J
        
        constraints = []
        for s in self.states:
            for a in self.actions:
                def constr_sa(x):
                    temp = 0
                    for s_ in self.states:
                        for a_ in self.actions:
                            temp += pi[s_,a_] * x[self._index(s_,a_)] * self.prob[s,a,s_]

                    return x[self.index(s,a)] - self.reward[s,a] - self.gamma * temp + x[self._index_xi(s,a)]
                
                constraints.append({
                    "type": "ineq",
                    "fun": constr_sa
                })

        def constr_xi(x):
            expr = 0
            for s in self.states:
                for a in self.actions:
                    expr += x[self._index_xi(s,a)] ** 2
            return self.eps - expr
                
        constraints.append({
            "type": "ineq",
            "fun": constr_xi
        })
        
        res = minimize(func, (2, 0), method='SLSQP', constraints=constraints)
    

    # Utility: calculate state-visit frequency.
    def _transit(self, distr, prob, pi):
        distr_new = np.zeros(shape=(self.num_states,), dtype=np.float64)
        for s in self.states:
            for a in self.actions:
                for s_ in self.states:
                    distr_new[s_] += distr[s] * pi[s,a] * prob[s,a,s_]
        
        return distr_new

    def visit_freq(self, pi, T):
        assert pi.shape == (self.num_states, self.num_actions)
        distr_cur = deepcopy(self.distr_init)
        g_t = 1
        d_pi = distr_cur
        for t in range(T):
            g_t *= self.gamma
            distr_cur = self._transit(distr_cur, self.prob, pi)
            d_pi += g_t * distr_cur
        
        d_pi *= 1-self.gamma
        return d_pi