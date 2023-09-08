import numpy as np
import random

from . import Agent


class SoftmaxPolicyGradientAgent(Agent):
    def __init__(self, env, eps, eta, T_est, thres):
        # Environment information.
        self.env            = env
        
        self.num_states     = env.num_states
        self.num_actions    = env.num_actions
        self.states         = env.states
        self.actions        = env.actions

        self.reward         = env.reward
        self.gamma          = env.gamma
        self.eps            = eps

        # Learning parameters.
        self.eta    = eta
        self.T_est  = T_est
        self.thres  = thres

        # Internal state.
        self.theta  = None
        self.pi     = None
    

    # Core functions.
    def _get_pi(self):
        pi = np.exp(self.theta)
        Z  = pi.sum(axis=1)[:,np.newaxis]
        self.pi = pi / Z
        return self.pi

    def reset(self, theta_init):
        assert theta_init.shape == (self.num_states, self.num_actions)
        self.theta = theta_init
        
        return self._get_pi()
    
    def update(self):
        pi = self._get_pi()

        Q_pi = self.env.robust_Q(pi=pi, eps=self.eps)

        V_pi = (Q_pi*pi).sum(axis=1)[:, np.newaxis]
        A_pi = Q_pi - V_pi

        d_pi = self.env.visit_freq(self.pi, T=self.T_est)[:, np.newaxis]
        grad = d_pi * pi * A_pi / (1-self.env.gamma)
        assert grad.shape == (self.num_states, self.num_actions)
        
        self.theta = self.theta + self.eta * grad
        return pi, {"Q_pi": Q_pi}

    def select_action(self, state):
        return random.choices(self.actions, weights=self.pi[int(state),:])[0]
    
    def save(self, path):
        np.save(path, self.pi)
    
    def load(self, path):
        self.pi = np.load(path)
    

    # Utility: projection onto the probability simplex.
    def _l2_project(self, r, a, b):
        # Implements l2-projection onto the simplex:
        #   min_y  ||r-x||^2
        #   s.t.   1^T x = 1
        #          a <= x <= b
        # REF: http://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html.
        assert (a.sum()<=1) and (b.sum()>=1) and np.all(a<=b), "Error: projection infeasible."
        lambdas = np.append(a-r, b-r)
        idx = np.argsort(lambdas)
        lambdas = lambdas[idx]
        active = np.cumsum((idx < r.size)*2 - 1)[:-1]
        diffs = np.diff(lambdas, n=1)
        totals = a.sum() + np.cumsum(active*diffs)
        i = np.searchsorted(totals, 1.0)
        lam = (1 - totals[i]) / active[i] + lambdas[i+1]
        return np.clip(r + lam, a, b)

    def _project(self, x):
        assert x.shape == (self.env.num_actions,)
        return self._l2_project(x, np.zeros_like(x), np.ones_like(x))