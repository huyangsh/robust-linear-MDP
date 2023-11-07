import numpy as np
import scipy as sp
from scipy.optimize import minimize
import random
from functools import partial
from copy import deepcopy
from math import exp, log

from . import Env


class TabularMDP(Env):
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


    # Utility: standard policy evaluation via DP.
    def DP_opt(self, thres):
        V = np.zeros(shape=(self.num_states,), dtype=np.float32)

        diff = thres + 1
        while diff > thres:
            V_prev = V
            V = np.zeros(shape=(self.num_states,), dtype=np.float32)

            for s in self.states:
                reward_max = None
                for a in self.actions:
                    V_pi_cum = 0
                    for s_ in self.states:
                        V_pi_cum += self.prob[s,a,s_] * V_prev[s_]

                    if reward_max is None:
                        reward_max = self.reward[s,a] + self.gamma*V_pi_cum
                    else:
                        reward_max = max(reward_max, self.reward[s,a] + self.gamma*V_pi_cum)
                
                V[s] = reward_max
            
            diff = np.linalg.norm(V - V_prev)
        
        return V
    
    def DP_pi(self, pi, thres=1e-6):
        assert pi.shape == (self.num_states, self.num_actions)
        V = np.zeros(shape=(self.num_states,), dtype=np.float32)

        diff = thres + 1
        while diff > thres:
            V_prev = V
            V = np.zeros(shape=(self.num_states,), dtype=np.float32)

            for s in self.states:
                for a in self.actions:
                    V_pi_cum = 0
                    for s_ in self.states:
                        V_pi_cum += self.prob[s,a,s_] * V_prev[s_]

                    V[s] += pi[s,a] * (self.reward[s,a] + self.gamma * V_pi_cum)
            
            diff = np.linalg.norm(V - V_prev)
        
        return V
    
    def V_to_Q(self, V):
        assert V.shape == (self.num_states,)
        Q = np.zeros(shape=(self.num_states, self.num_actions), dtype=np.float32)
        for s in self.states:
            for a in self.actions:
                V_pi_cum = 0
                for s_ in self.states:
                    V_pi_cum += self.prob[s,a,s_] * V[s_]

                Q[s,a] = self.reward[s,a] + self.gamma * V_pi_cum
        
        return Q
    
    def DP_Q(self, pi):
        return self.V_to_Q(self.DP_pi(pi=pi))
    
    def Q_to_pi(self, Q):
        assert pi.shape == (self.num_states, self.num_actions)
        indices = np.argmax(Q, axis=1)[:,np.newaxis]
        pi = np.zeros(shape=(self.num_states, self.num_actions), dtype=np.float32)
        np.put_along_axis(pi, indices=indices, values=1, axis=1)
        return pi
    
    def get_opt_pi(self):
        V_opt  = self.DP_opt(thres=1e-6)
        Q_opt  = self.V_to_Q(V_opt)
        pi_opt = self.Q_to_pi(Q_opt)
        return pi_opt


    # Utility: robust policy evaluation.
    def _index_Q(self, s, a):
        return s*self.num_actions + a
    
    def _index_xi(self, s, a):
        return (self.num_states+s)*self.num_actions + a
    
    def _index_delta(self, s, a, s_):
        return self.num_states*self.num_actions + s*self.num_actions*self.num_states + a*self.num_states + s_

    def robust_Q(self, pi, eps=0):
        def func(x):
            J = 0
            for s in self.states:
                for a in self.actions:
                    J += self.distr_init[s] * pi[s,a] * x[self._index_Q(s,a)]
            return J
        
        def constr_sa(s, a, x):
            tot = 0
            for s_ in self.states:
                for a_ in self.actions:
                    tot += pi[s_,a_] * x[self._index_Q(s_,a_)] * self.prob[s,a,s_]

            return x[self._index_Q(s,a)] - self.reward[s,a] - self.gamma * tot + x[self._index_xi(s,a)]
        
        def constr_xi(x):
            tot = 0
            for s in self.states:
                for a in self.actions:
                    tot += x[self._index_xi(s,a)] ** 2
            return eps - tot
        
        def box_lower_xi(s, a, x):
            return eps + x[self._index_xi(s,a)]
        def box_upper_xi(s, a, x):
            return -x[self._index_xi(s,a)]

        constraints = []
        for s in self.states:
            for a in self.actions:
                constraints.append({
                    "type": "ineq",
                    "fun": partial(constr_sa, s, a)
                })    
        """constraints.append({
            "type": "ineq",
            "fun": constr_xi
        })"""
        for s in self.states:
            for a in self.actions:
                constraints.append({
                    "type": "ineq",
                    "fun": partial(box_lower_xi, s, a)
                })
                constraints.append({
                    "type": "ineq",
                    "fun": partial(box_upper_xi, s, a)
                })
        
        res = minimize(
            func, np.zeros(shape=(2*self.num_states*self.num_actions,)), constraints=constraints,
            method='SLSQP', tol=1e-8, options={"maxiter": 1000, "ftol": 1e-8, "eps": 1e-8}
        )

        return res.x[:self.num_states*self.num_actions].reshape((self.num_states, self.num_actions))


    def robust_prob_Q(self, pi, eps=0):
        def func(x):
            J = 0
            for s in self.states:
                for a in self.actions:
                    J += self.distr_init[s] * pi[s,a] * x[self._index_Q(s,a)]
            return J
        
        def constr_sa(s, a, x):
            tot = 0
            for s_ in self.states:
                for a_ in self.actions:
                    tot += pi[s_,a_] * x[self._index_Q(s_,a_)] * (self.prob[s,a,s_] + x[self._index_delta(s,a,s_)])

            return x[self._index_Q(s,a)] - self.reward[s,a] - self.gamma * tot 
        
        def constr_delta(s, a, x):
            tot = 0
            for s_ in self.states:
                tot += x[self._index_delta(s,a,s_)]
            return tot
        
        def box_lower_delta(s, a, s_, x):
            return self.prob[s,a,s_] + x[self._index_delta(s,a,s_)]
        def box_upper_delta(s, a, s_, x):
            return 1 - self.prob[s,a,s_] - x[self._index_delta(s,a,s_)]

        constraints = []
        for s in self.states:
            for a in self.actions:
                constraints.append({
                    "type": "ineq",
                    "fun": partial(constr_sa, s, a)
                })    
                constraints.append({
                    "type": "eq",
                    "fun": partial(constr_delta, s, a)
                })
        for s in self.states:
            for a in self.actions:
                for s_ in self.states:
                    constraints.append({
                        "type": "ineq",
                        "fun": partial(box_lower_delta, s, a, s_)
                    })
                    constraints.append({
                        "type": "ineq",
                        "fun": partial(box_upper_delta, s, a, s_)
                    })
        
        res = minimize(
            func, np.zeros(shape=((1+self.num_states)*self.num_states*self.num_actions,)), constraints=constraints,
            method='SLSQP', tol=1e-8, options={"maxiter": 1000, "ftol": 1e-8, "eps": 1e-8}
        )
        print(res.message)

        return res.x[:self.num_states*self.num_actions].reshape((self.num_states, self.num_actions))

    # Utility: calculate state-visit frequency.
    def _transit(self, distr, prob, pi):
        distr_new = np.zeros(shape=(self.num_states,), dtype=np.float32)
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