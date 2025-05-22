# SDP problem to compute robust covariance terms from the paper DRO LQG by Daniel Kuhn.

import cvxpy as cp
import numpy as np
import scipy as sp

n = 1 #dimension of the states x
m = 1 #dimension of thz inputs u
p = 1 #dimension of the outputs y


# --- Simulation Parameters ---
time_horizon = 30                # time horizon
dt = 1                         # time step
T = int(time_horizon / dt)       # number of steps

# --- Model Parameters ---
theta = 0.15          # mean-reversion rate
mu = 130.0            # target portfolio value
sigma_w = 2.0         # noise strength for the system
sigma_v = 2.0         # noise strength for the observation
sigma_x0 = 2.0       # noise strength for the initial condition


np.random.seed(0)
# Random system matrices for now (to be replaced with real data)
# A = np.eye(n) 
# B = 0.1*np.random.randn(n, m)

# Model parameters for the OU process
A = np.array([[1 - theta * dt]])  
B = np.array([[-dt]])             
C = np.eye(p, n) + 0.1*np.random.randn(p, n)

# Variables 
X0 = cp.Variable((n, n), PSD=True)
W_list = [cp.Variable((n, n), PSD=True) for _ in range(T)] # We add the X0 later
V_list = [cp.Variable((p, p), PSD=True) for _ in range(T)]
F = cp.Variable((T * m, T * m), PSD=True)

# Wasserstein slack variables
Ex0 = cp.Variable((n, n),PSD=True)
Ew_list = [cp.Variable((n, n),PSD=True) for _ in range(T)]
Ev_list = [cp.Variable((p, p),PSD=True) for _ in range(T)]


# Data definition (to change)
X0_hat = sigma_x0**2 * np.eye(n)
W_hat_list = [sigma_w**2 * dt * np.eye(n) for _ in range(T)] # Variance matrices for W
V_hat_list = [sigma_v**2 * np.eye(p) for _ in range(T)] # Variance matrices for V
rho_x0 = 1.0
rho_w = 1.0
rho_v = 1.0


Q = np.eye(n*(T+1))  # Had t oadjust the dimensions of Q to match the G matrix but not convinced. 
R = np.eye(p*T) 

# Build G matrix (n(T+1) x n(T+1)) 
G = np.zeros((n*(T+1), n*(T+1)))

for t in range(T+1):
    for s in range(t):
        A_prod = np.linalg.matrix_power(A, t-s)  # A^(t-s)
        G[n*t:n*(t+1), n*s:n*(s+1)] = A_prod
    G[n*t:n*(t+1), n*t:n*(t+1)] = np.eye(n)


# Build H matrix (n(T+1) x mT) 
H = np.zeros((n*(T+1), m*T))

for t in range(1, T+1):
    for s in range(t):
        A_prod = np.linalg.matrix_power(A, t-s-1) #if (t-s-1) >= 0 else np.eye(n)
        H[n*t:n*(t+1), m*s:m*(s+1)] = A_prod @ B


# Build C matrix (pT x n(T+1))
C_blocks = []
for t in range(T):
    row = [np.zeros((p, n)) for _ in range(T+1)] # Initialize row with zeros
    row[t] = C # Only the t-th block is non-zero
    C_blocks.append(row)

C_big = np.block(C_blocks)

# Build D matrix
D = C_big @ G

# Build W and V block-diagonal matrices
W_blocks = [X0] + W_list
W = cp.bmat([[W_blocks[i] if i==j else np.zeros((n, n)) for j in range(T+1)] for i in range(T+1)])

V = cp.bmat([[V_list[i] if i==j else np.zeros((p, p)) for j in range(T)] for i in range(T)])

# Build M matrix (mT x pT)
M = cp.bmat([[cp.Variable([m,p]) if i<j else np.zeros((m, p)) for j in range(T)] for i in range(T)])

# Objective: maximize Tr(G^T Q G W) - Tr(F (R + H^T Q H)^-1)
objective = cp.Maximize(cp.trace(G.T @ Q @ G @ W) - cp.trace(F @ np.linalg.inv(R + H.T @ Q @ H)))

# Constraints
constraints = []

# Wasserstein constraints for X0
min_eigenvalue_X0 = sp.linalg.eigvalsh(X0_hat)[0]
constraints += [X0 - min_eigenvalue_X0 *np.eye(n) >> 0]  

sqrt_X0_hat = sp.linalg.sqrtm(X0_hat)
constraints += [cp.trace(X0 + X0_hat - 2*Ex0) <= rho_x0**2]
constraints += [
    cp.bmat([
        [sqrt_X0_hat @ X0 @ sqrt_X0_hat, Ex0],
        [Ex0, np.eye(n)]
    ]) >> 0
]

# Wasserstein constraints for W_t
for t in range(T):
    min_eigenvalue_W = sp.linalg.eigvalsh(W_hat_list[t])[0]
    constraints += [W_list[t] - min_eigenvalue_W *np.eye(n) >> 0] 

    sqrt_W_hat = sp.linalg.sqrtm(W_hat_list[t])
    constraints += [cp.trace(W_list[t] + W_hat_list[t] - 2*Ew_list[t]) <= rho_w**2]  # W[t+1] is the W_t because W[0] corresponds to X0
    constraints += [
        cp.bmat([
            [sqrt_W_hat @ W_list[t] @ sqrt_W_hat, Ew_list[t]],
            [Ew_list[t], np.eye(n)]
        ]) >> 0
    ]

# Wasserstein constraints for V_t
for t in range(T):
    min_eigenvalue_V = sp.linalg.eigvalsh(V_hat_list[t])[0]
    constraints += [V_list[t] - min_eigenvalue_V *np.eye(p) >> 0]  

    sqrt_V_hat = sp.linalg.sqrtm(V_hat_list[t])
    constraints += [cp.trace(V_list[t] + V_hat_list[t] - 2*Ev_list[t]) <= rho_v**2]
    constraints += [
        cp.bmat([
            [sqrt_V_hat @ V_list[t] @ sqrt_V_hat, Ev_list[t]],
            [Ev_list[t], np.eye(p)]
        ]) >> 0
    ]

# Big semidefinite constraint
constraints += [
    cp.bmat([
        [F, H.T @ Q @ G @ W @ D.T + M/2],
        [(H.T @ Q @ G @ W @ D.T + M/2).T, D @ W @ D.T + V]
    ]) >> 0
]



# Define problem
problem = cp.Problem(objective, constraints)

# Solve
problem.solve(solver=cp.MOSEK)

# Print results
print("Status:", problem.status)
print("\nOptimal value:", problem.value)
print("Optimal X0:\n", X0.value)
print("First W0:\n", W_list[0].value)
print("First V0:\n", V_list[0].value)


