import numpy as np
import random

from . import Agent


class LinearPolicyGradientAgent(Agent):
    def __init__(self, env, eta, T_est, T_Q, thres, use_dual=False):
        # Environment information.
        self.env            = env
        
        self.num_states     = env.num_states
        self.num_actions    = env.num_actions
        self.states         = env.states
        self.actions        = env.actions

        self.reward         = env.reward
        self.gamma          = env.gamma

        self.use_dual       = use_dual

        # Learning parameters.
        self.eta    = eta
        self.T_est  = T_est
        self.T_Q    = T_Q
        self.thres  = thres

        # Internal state.
        self.pi      = None

        self.use_NPG = True
        self.theta   = np.zeros(shape=(self.num_states, self.num_actions), dtype=np.float32)
    

    # Core functions.
    def reset(self, pi_init):
        assert pi_init.shape == (self.num_states, self.num_actions)
        self.pi = pi_init

        if self.use_NPG:
            self.theta = np.log(self.pi)
        
        return self.pi
    
    def update(self):
        if self.use_dual:
            Q_pi, phi_pi = self.env.robust_Q_dual(pi=self.pi, T=self.T_Q)
        else:
            Q_pi, phi_pi = self.env.robust_Q(pi=self.pi, T=self.T_Q)

        # d_pi = self.env.visit_freq(self.pi, T=self.T_est)[:, np.newaxis]
        if self.use_NPG:
            # Natural policy gradient via soft-max parametrization.
            V_pi = np.sum(Q_pi*self.pi, axis=1)[:, np.newaxis]
            self.theta = self.theta + self.eta / (1-self.gamma) * (Q_pi - V_pi)

            exp_theta = np.exp(self.theta)
            exp_theta_sum = np.sum(exp_theta, axis=1)[:, np.newaxis]
            self.pi = np.divide(exp_theta, exp_theta_sum)
        else:
            # Direct-parametrization policy gradient.
            d_pi = self.env.robust_visit_freq(pi=self.pi, phi=phi_pi, T=self.T_est)[:, np.newaxis]

            grad = Q_pi * d_pi / (1-self.env.gamma)
            assert grad.shape == (self.num_states, self.num_actions)
            
            self.pi = self.pi + self.eta * grad
            for s in self.env.states:
                self.pi[s, :] = self._project(self.pi[s, :])
        
        return self.pi, {"Q_pi": Q_pi, "phi_pi": phi_pi}

    def select_action(self, state):
        return random.choices(self.actions, weights=self.pi[int(state),:])[0]
    
    def save(self, path):
        np.save(path, self.pi)
    
    def load(self, path):
        self.pi = np.load(path)
    

    # Utility: projection onto the probability simplex.
    def _l2_project_bounded(self, v, a, b):
        # Implements l2-projection onto the simplex with bounds:
        #   min_x  ||v-x||^2
        #   s.t.   1^T x = 1
        #          a <= x <= b
        # REF: http://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html.
        assert (a.sum()<=1) and (b.sum()>=1) and np.all(a<=b), "Error: projection infeasible."
        lambdas = np.append(a-v, b-v)
        idx = np.argsort(lambdas)
        lambdas = lambdas[idx]
        active = np.cumsum((idx < v.size)*2 - 1)[:-1]
        diffs = np.diff(lambdas, n=1)
        totals = a.sum() + np.cumsum(active*diffs)
        i = np.searchsorted(totals, 1.0)
        lam = (1 - totals[i]) / active[i] + lambdas[i+1]
        return np.clip(v + lam, a, b)
    
    def _l2_project(self, v):
        # Implements l2-projection onto the simplex.
        #   min_x  ||v-x||^2
        #   s.t.   1^T x = 1
        #          x >= 0
        # REF: https://arxiv.org/abs/1309.1541 and https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf.
        # WARNING: numerical issues exist. Be sure to add numbers in similar magnitude first.
        n = v.shape[0]
        u = np.sort(v)[::-1]
        u_sum = np.cumsum(u)
        ind = np.arange(n) + 1
        cond = (u - u_sum/ind + 1/ind) > 0
        rho = ind[cond][-1]
        lamda_1 = -u_sum[cond][-1] / float(rho)
        lamda_2 = 1 / float(rho)
        return np.maximum(v + lamda_1 + lamda_2, 0)

    def _project(self, x):
        assert x.shape == (self.env.num_actions,)
        # return self.projection_simplex_sort(x,)
        return self._l2_project(x)