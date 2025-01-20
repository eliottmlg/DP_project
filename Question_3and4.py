# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:32:04 2025

@author: eliot
"""

#==============================================================================
# Dynamic Programming - Project
# QUESTIONS 3, 4
#==============================================================================


import os
import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
import pandas as pd


def iterative_policy_function(w, policy, grid, β, delta, u, f, updated_w = None):
    w_func = lambda x: np.interp(x, grid, w)  # Interpolates value function
    if updated_w is None:
        updated_w = np.empty_like(w)

    for i, h in enumerate(grid):
        # Compute next period's human capital
        hprime = (1 - delta)*h + 1 - policy[i]
        # Compute the updated value using the Bellman equation
        updated_w[i] = u(f(h, policy[i])) + (β * w_func(hprime))
    return updated_w


def solve_optimal_policy(initial_w, policy, grid, β, delta, u, f, tol=1e-4, max_iter=500):
    w = np.empty(len(grid))
    w[:] = initial_w  # Set initial condition
    error = tol + 1
    i = 0

    # == Create storage array for bellman_operator. Reduces  memory allocation and speeds code up == #
    updated_w = np.empty(len(grid))

    # Iterate to find solution
    while error > tol and i < max_iter:
        w_new = iterative_policy_function(w, policy, grid, β, delta, u, f, updated_w)

        error = np.max(np.abs(w_new - w))
        w[:] = w_new
        i += 1
        print("Iteration "+str(i)+'\n Error is '+str(error)+'\n') if i % 50 == 0 or error < tol else None

    return [w]


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
        return (c**(1-self.sigma)-1)/(1-self.sigma)

    def f(self, h,l):
        " Deterministic part of f function.  "
        return h**self.α * l
    
    
# Creation of the model
ces = CES_OG()
# == Unpack parameters / functions for convenience == #
α, β, sigma, delta = ces.α, ces.β, ces.sigma, ces.delta


### Setup of the grid
grid_max = 5         # Largest grid point
grid_size = 4000     # Number of grid points
grid = np.linspace(1e-5, grid_max, grid_size)
arb_policy = 0.5 * grid  # Arbitrary policy for labor allocation


# Initial guess for the value function
guess_w = ces.u(ces.f(grid, arb_policy))

# Computation of the value function
solve = solve_optimal_policy(guess_w, arb_policy, grid, β, delta, u=ces.u,
                               f=ces.f, tol=1e-4, max_iter=500)

value_approx = solve[0]


#==============================================================================
# Plotting value function
#==============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(grid, value_approx, lw=2, alpha=0.6, label='approximate value function')
ax.plot(grid, guess_w, lw=2, alpha=0.6, label='guessed value function')
ax.set_xlabel('c')  # i think this is wrong: grid = human capital is on the x-axis
ax.set_ylabel('v')
ax.legend(loc='lower right')
plt.show()

    
# Define e-greedy policy improvement
def improve_policy_epsilon_greedy(w, grid, β, delta, u, f, epsilon=0.01):
    """

    Returns:
        Tw: Updated value function.
        Tpolicy: Updated policy function.
    """
    w_func = lambda x: np.interp(x, grid, w)
    Tw = np.empty_like(w)
    l_opt = np.empty_like(w)

    for i, h in enumerate(grid):
        def objective(l, h=h):
            hprime = (1 - delta)*h + 1 - l
            return - u(f(h, l)) - (β * w_func(hprime))
        l_star = fminbound(objective, 0, 1)

        if np.random.rand() < epsilon:  # Exploration
            l_star = np.random.uniform(1e-10, 1)  # Random labor allocation
        else:  # Exploitation
            l_star = fminbound(objective, 0, 1)  # Find optimal labor allocation

        l_opt[i] = l_star
        Tw[i] = -objective(l_star)  # Reverse sign to get max value

    return Tw, l_opt


# Iterative solution with e-greedy policy improvement
def solve_with_epsilon_greedy(initial_policy, grid, β, delta, u, f, epsilon=0.01, tol=1e-5, max_iter=100):
    """
    Solve the model using e-greedy policy improvement.
    """
    policy = initial_policy
    value_function = np.zeros_like(grid)
    w_history = []  # To store value functions at each iteration
    error = tol + 1
    iteration = 0

    while error > tol and iteration < max_iter:
        # Evaluate the value function for the current policy
        new_value_function = iterative_policy_function(
            value_function, policy, grid, β, delta, u, f
        )

        # Improve the policy using e-greedy exploration
        new_value_function, new_policy = improve_policy_epsilon_greedy(
            new_value_function, grid, β, delta, u, f, epsilon
        )

        # Check convergence
        error = np.max(np.abs(new_value_function - value_function))
        value_function[:] = new_value_function
        policy[:] = new_policy
        w_history.append(value_function.copy())  # Save the value function for plotting
        iteration += 1

        print(f"Iteration {iteration}, Error: {error}")  if iteration % 1 == 0 or error < tol else None

    return value_function, policy, w_history


### Setup of the grid
grid_max = 5         # Largest grid point
grid_size = 4000     # Number of grid points
grid = np.linspace(1e-5, grid_max, grid_size)
arb_policy = 0.5 * grid  # Arbitrary policy for labor allocation

# Solve using e-greedy policy improvement
value_function, optimal_policy, w_history = solve_with_epsilon_greedy(
    arb_policy, grid, β, delta, ces.u, ces.f, epsilon=0.01
)

# Plot the value functions across iterations
w_matrix = np.array(w_history)
fig, ax = plt.subplots(figsize=(9, 5))

# Set y-axis limits to span the range of value functions
ax.set_ylim(min(w_matrix[-1]) - 1, max(w_matrix[-1]) + 1)

# Plot each value function from the iterations
for i, w_iter in enumerate(w_matrix[0:4,:]):
    ax.plot(grid, w_iter, lw=2, alpha=0.6, label=f'Iteration {i+1}' if i < 10 else None)

ax.plot(grid, w_matrix[-1,:], lw=2, alpha=0.6, label=f'Final iteration' if i < 10 else None)

# Label the axes
ax.set_xlabel('h')
ax.set_ylabel('v')
ax.legend(loc='lower right', ncol=2)

# Display the plot
plt.title('Value function across iterations (ε-greedy)')
plt.show()



