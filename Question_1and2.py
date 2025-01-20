# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:55:10 2025

@author: eliot
"""

#==============================================================================
# Dynamic Programming - Project
# QUESTIONS 1, 2
#==============================================================================

import os
import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
import pandas as pd


def bellman_operator(w, grid, β, delta, u, f, Tw=None, compute_policy=0):
    # === Apply linear interpolation to w === #
    w_func = lambda x: np.interp(x, grid, w)

    # == Initialize Tw if necessary == #
    if Tw is None:
        Tw = np.empty_like(w)

    if compute_policy:
        σ = np.empty_like(w)
        c_opt = np.empty_like(w)
        l_opt = np.empty_like(w)

    # == set Tw[i] = max_c { u(c) + β E w(f(y  - c) z)} == #
    for i, h in enumerate(grid):
        def objective(l, h=h):
            hprime = (1 - delta)*h + 1 - l
            return - u(f(h, l)) - (β * w_func(hprime))
        l_star = fminbound(objective, 0, 1)
            
        if compute_policy:
            σ[i] = (1-delta) * h + 1 - l_star #h_(t+1) as a function of h_t
            c_opt[i] = f(h,l_star)
            l_opt[i] = l_star
        Tw[i] = - objective(l_star)
        

    if compute_policy:
        return Tw, σ, c_opt, l_opt
    else:
        return Tw



def solve_optgrowth(initial_w, grid, β, delta, u, f, tol=1e-4, max_iter=500):

    w = initial_w  # Set initial condition
    error = tol + 1
    i = 0

    # == Create storage array for bellman_operator. Reduces  memory
    # allocation and speeds code up == #
    Tw = np.empty(len(grid))

    # Iterate to find solution
    while error > tol and i < max_iter:
        w_new = bellman_operator(w,
                                 grid,
                                 β,
                                 delta,
                                 u,
                                 f,
                                 Tw)
        error = np.max(np.abs(w_new - w))
        w[:] = w_new
        i += 1
        print("Iteration "+str(i)+'\n Error is '+str(error)+'\n') if i % 50 == 0 or error < tol else None

        
    # Computes policy
    w, policy, c_opt, l_opt = bellman_operator(w,
                             grid,
                             β,
                             delta,
                             u,
                             f,
                             Tw,
                             compute_policy=1)
 
    return [w, policy, c_opt, l_opt]



class CES_OG:
    """
    Constant elasticity of substitution optimal growth model so that

        c = f(h,l) = h^α * l


    The class holds parameters and true value and policy functions.
    """

    def __init__(self, α=0.4, β=0.9, sigma=0.9, delta=0.05):

        self.α, self.β, self.sigma, self.delta = α, β, sigma, delta

    def u(self, c):
        " Utility "
        return (c**(1-self.sigma))/(1-self.sigma)

    def f(self, h,l):
        " consumption function.  "
        return h**self.α * l
    
    
# Creation of the model
ces = CES_OG()
# == Unpack parameters / functions for convenience == #
α, β, sigma, delta = ces.α, ces.β, ces.sigma, ces.delta


### Setup of the grid
grid_max = 5         # Largest grid point
grid_size = 4000     # Number of grid points
grid = np.linspace(1e-5, grid_max, grid_size)

# Initial conditions and shocks
initial_w = 5 * np.log(grid)


# Computation of the value function
solve = solve_optgrowth(initial_w, grid, β, delta, u=ces.u,
                               f=ces.f, tol=1e-4, max_iter=500)

value_approx = solve[0]
policy_function = solve[1]
c_opt = solve[2]
l_opt = solve[3]

#==============================================================================
# Plotting value function
#==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_ylim(min(value_approx), max(value_approx))
ax.plot(grid, value_approx, lw=2, alpha=0.6, label='approximate value function')
ax.set_xlabel('h')
ax.set_ylabel('v')
ax.legend(loc='lower right')
plt.show()


#==============================================================================
# Plotting Policy function
#==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_ylim(min(policy_function), max(policy_function))
ax.plot(grid, policy_function, lw=2, alpha=0.6, label='approximate policy function')

# 45° line
ax.plot(grid, grid, lw=2, alpha=0.6, label='45 degrees line')

ax.set_xlabel('h_t')
ax.set_ylabel('h_(t+1)')
ax.legend(loc='lower right')
plt.show()

#==============================================================================
# Plotting Consumption
#==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_ylim(min(c_opt), max(c_opt))
ax.plot(grid, c_opt, lw=2, alpha=0.6, label='approximate consumption')

# 45° line
ax.plot(grid, grid, lw=2, alpha=0.6, label='45 degrees line')

ax.set_xlabel('h_t')
ax.set_ylabel('c_t')
ax.legend(loc='lower right')
plt.show()


#==============================================================================
# Plotting labour supply
#==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_ylim(0,1.1)
ax.plot(grid, l_opt, lw=2, alpha=0.6, label='approximate labour supply')
ax.plot(grid, (1-l_opt), lw=2, alpha=0.6, label='approximate learning')

# 45° line
ax.plot(grid, grid, lw=2, alpha=0.6, label='45 degrees line')

ax.set_xlabel('h_t')
ax.set_ylabel('l_t')
ax.legend(loc='lower right')
plt.show()


#==============================================================================
# Finding steady state
#==============================================================================

# Creation of the model
ces = CES_OG()
# == Unpack parameters / functions for convenience == #
α, β, sigma, delta = ces.α, ces.β, ces.sigma, ces.delta

ss = np.absolute(grid - policy_function)
ss = pd.DataFrame(ss)
index_min = ss.idxmin() 
ss_h = policy_function[index_min] # SS of h
ss_l = 1-delta*ss_h # l
ss_c = ces.f(ss_h,ss_l) # c
