import numpy as np
import scipy as sp
from scipy.optimize import minimize
import random
from functools import partial
from copy import deepcopy
from math import exp, log

from . import Env


class LinearMDP(Env):
    def __init__(self, distr_init, phi, theta, mu, gamma, eps_phi=0, eps_theta=0, eps_mu=0, thres=1e-5):
        self.num_states, self.num_actions, self.dim_feature  = phi.shape
        assert distr_init.shape == (self.num_states,)
        assert theta.shape == (self.dim_feature,)
        assert mu.shape == (self.num_states, self.dim_feature)
        assert gamma <= 1
        
        self.states      = np.arange(self.num_states)
        self.actions     = np.arange(self.num_actions)
        self.dim_state   = 1
        self.dim_action  = 1

        self.distr_init  = distr_init

        self.phi    = phi
        self.theta  = theta
        self.mu     = mu

        self.reward = phi @ theta
        self.prob   = np.squeeze(mu @ phi[:,:,:,np.newaxis])

        self.gamma  = gamma
        self.thres  = thres

        # Perturbation radii.
        self.eps_phi     = eps_phi
        self.eps_theta   = eps_theta
        self.eps_mu      = eps_mu


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
    
    def Q_to_V(self, Q, pi):
        assert Q.shape == (self.num_states, self.num_actions)
        assert pi.shape == (self.num_states, self.num_actions)
        return np.sum(Q*pi, axis=1)
    
    def Q_to_pi(self, Q):
        assert Q.shape == (self.num_states, self.num_actions)
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
    def robust_Q(self, pi, T):
        # Note that the following does not change across iterations.
        const_eta = np.sum((self.distr_init[:,np.newaxis] * pi)[:,:,np.newaxis] * self.phi, axis=(0,1))
        # The perturbation radius is still eps_phi.

        V_t = np.zeros(shape=(self.num_states,), dtype=np.float32)
        for t in range(T):
            # Note that the following does change across iterations.
            const_xi = self.theta + self.mu.T @ V_t
            V_t_clipped = np.minimum(abs(V_t), np.ones(shape=(self.num_states,)) / self.gamma)
            eps_xi = self.eps_theta + self.gamma* np.sum(V_t_clipped) * self.eps_mu

            # Optimization step: use the reduced form.
            def func(x):
                return const_eta @ x - self.eps_phi * np.linalg.norm(const_xi + x)
            
            def constr_ball(x):
                return eps_xi**2 - x @ x

            constraints = [{
                "type": "ineq",
                "fun": constr_ball
            }]
            res = minimize(
                func, np.zeros(shape=(self.dim_feature,)), constraints=constraints,
                method='SLSQP', tol=1e-8, options={"maxiter": 1000, "ftol": 1e-8, "eps": 1e-8}
            )

            xi_t = res.x
            eta_t = - self.eps_phi * xi_t / np.linalg.norm(xi_t)

            # Update omega and V.
            omega_t = self.theta + self.gamma * self.mu.T @ V_t + xi_t
            phi_t   = self.phi + eta_t
            Q_t     = phi_t @ omega_t
            V_t     = np.sum(Q_t * pi, axis=1)
            assert V_t.shape[0] == self.num_states

        return Q_t, phi_t
    
    def robust_Q_dual(self, pi, T):
        # Note that the following does not change across iterations.
        const_eta = np.sum((self.distr_init[:,np.newaxis] * pi)[:,:,np.newaxis] * self.phi, axis=(0,1))
        # The perturbation radius is still eps_phi.

        V_t = np.zeros(shape=(self.num_states,), dtype=np.float32)
        for t in range(T):
            # Note that the following does change across iterations.
            const_xi = self.theta + self.mu.T @ V_t
            V_t_clipped = np.minimum(abs(V_t), np.ones(shape=(self.num_states,)) / self.gamma)
            eps_xi = self.eps_theta + self.gamma* np.sum(V_t_clipped) * self.eps_mu

            # Optimization step: use the reduced form.
            """
            def func(x, alpha):
                obj = (const_xi + x[:self.dim_feature]) @ (const_eta + x[self.dim_feature:])
                obj += alpha * x[:self.dim_feature] @ x[:self.dim_feature]
                obj += alpha * x[self.dim_feature:] @ x[self.dim_feature:]
                return obj
            
            alpha = max(200, 200 * np.linalg.norm(const_xi) * np.linalg.norm(const_eta))
            res = minimize(
                partial(func, alpha=alpha), np.zeros(shape=(2*self.dim_feature,)), 
                method='BFGS', tol=1e-8, options={"maxiter": 1000, "gtol": 1e-8, "eps": 1e-8}
            )

            xi_t = res.x[:self.dim_feature]
            eta_t = res.x[self.dim_feature:]
            """

            alpha = 20 * np.linalg.norm(const_xi) * np.linalg.norm(const_eta)
            xi_t = - (2*alpha*const_eta - const_xi) / (4*alpha*alpha-1)
            eta_t = - (2*alpha*const_xi - const_eta) / (4*alpha*alpha-1)

            # Update omega and V.
            omega_t = self.theta + self.gamma * self.mu.T @ V_t + xi_t
            phi_t   = self.phi + eta_t
            Q_t     = phi_t @ omega_t
            V_t     = np.sum(Q_t * pi, axis=1)
            assert V_t.shape[0] == self.num_states

        return Q_t, phi_t


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
    
    def robust_visit_freq(self, pi, phi, T):
        assert pi.shape == (self.num_states, self.num_actions)
        prob_robust = np.squeeze(self.mu @ phi[:,:,:,np.newaxis])

        distr_cur = deepcopy(self.distr_init)
        g_t = 1
        d_pi = distr_cur
        for t in range(T):
            g_t *= self.gamma
            distr_cur = self._transit(distr_cur, prob_robust, pi)
            d_pi += g_t * distr_cur
        
        d_pi *= 1-self.gamma
        return d_pi