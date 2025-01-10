# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 20:17:52 2025

@author: eliot
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:37:30 2018
@author: Bruno
"""

#==============================================================================
# Recursive Methods - Optimal Growth
#==============================================================================

import os
import numpy as np
from scipy.optimize import fminbound


def bellman_operator(w, grid, β, u, f, Tw=None, compute_policy=0):
    # === Apply linear interpolation to w === #
    w_func = lambda x: np.interp(x, grid, w)

    # == Initialize Tw if necessary == #
    if Tw is None:
        Tw = np.empty_like(w)

    if compute_policy:
        σ = np.empty_like(w)

    # == set Tw[i] = max_c { u(c) + β E w(f(y  - c) z)} == #
    for i, y in enumerate(grid):
        def objective(c, y=y):
            # return #Your code goes here
            return - u(c) - β * w_func(f(y - c))
        c_star = fminbound(objective, 1e-10, y)
        if compute_policy:
            # σ[i] = #Your code goes here #y_(t+1) as a function of y_t
            σ[i] = f(y-c_star) #y_(t+1) as a function of y_t
        # Tw[i] = #Your code goes here
        Tw[i] = - objective(c_star)

    if compute_policy:
        return Tw, σ
    else:
        return Tw



def solve_optgrowth(initial_w, grid, β, u, f, tol=1e-4, max_iter=500):

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
                                 u,
                                 f,
                                 Tw)
        # error = #Your code goes here 
        error = np.max(np.abs(w_new - w))
        w[:] = w_new
        i += 1
        print("Iteration "+str(i)+'\n Error is '+str(error)+'\n') if i % 50 == 0 or error < tol else None

        
    # Computes policy
    policy = bellman_operator(w,
                             grid,
                             β,
                             u,
                             f,
                             Tw,
                             compute_policy=1)[1]
 
    return [w, policy]



class CES_OG:
    """
    Constant elasticity of substitution optimal growth model so that

        y = f(k) = k^α


    The class holds parameters and true value and policy functions.
    """

    def __init__(self, α=0.4, β=0.96, sigma=0.9):

        self.α, self.β, self.sigma = α, β, sigma 

    def u(self, c):
        " Utility "
        return (c**(1-self.sigma)-1)/(1-self.sigma)

    def f(self, k):
        " Deterministic part of production function.  "
        return k**self.α
    
    
    
    
    