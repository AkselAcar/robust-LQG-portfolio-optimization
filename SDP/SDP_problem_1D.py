# SDP to find the robust covarince terms in one dimension 

import cvxpy as cp
import numpy as np

# ---------------------
# SETTINGS
# ---------------------
n = 2   # State dimension
m = 1   # Control input dimension
p = 2   # Observation dimension
T = 3   # Time horizon
# ---------------------

np.random.seed(0)

# Create one system matrix each (time-invariant)
A = np.eye(n) + 0.1*np.random.randn(n, n)
B = 0.1*np.random.randn(n, m)
C = np.eye(p, n) + 0.1*np.random.randn(p, n)

# Optimization variables
X0 = cp.Variable((n, n), PSD=True)
W_list = [cp.Variable((n, n), PSD=True) for _ in range(T)]
V_list = [cp.Variable((p, p), PSD=True) for _ in range(T)]

F = cp.Variable((m*T, m*T), PSD=True)
M = cp.Variable((m*T, p*T))

# Wasserstein slack variables
Ex0 = cp.Variable((n, n))
Ew_list = [cp.Variable((n, n)) for _ in range(T)]
Ev_list = [cp.Variable((p, p)) for _ in range(T)]

# Reference matrices
X0_hat = np.eye(n)
W_hat_list = [np.eye(n) for _ in range(T)]
V_hat_list = [np.eye(p) for _ in range(T)]

# Wasserstein radii
rho_x0 = 1.0
rho_w = 1.0
rho_v = 1.0


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
        A_prod = np.linalg.matrix_power(A, t-s-1) if (t-s-1) >= 0 else np.eye(n)
        H[n*t:n*(t+1), m*s:m*(s+1)] = A_prod @ B


# Build C matrix (pT x n(T+1))
C_blocks = []
for t in range(T):
    row = [np.zeros((p, n)) for _ in range(T+1)] # Initialize row with zeros
    row[t] = C # Only the t-th block is non-zero
    C_blocks.append(row)

C_big = np.block(C_blocks)

# ---------------------
# Build D matrix
# ---------------------
D = C_big @ G

# ---------------------
# Build W and V block-diagonal matrices
# ---------------------
W_blocks = [X0] + W_list
W = cp.bmat([[W_blocks[i] if i==j else None for j in range(T+1)] for i in range(T+1)])

V = cp.bmat([[V_list[i] if i==j else None for j in range(T)] for i in range(T)])

# ---------------------
# Objective
# ---------------------
objective = cp.Maximize(cp.trace(G.T @ np.eye(n*(T+1)) @ G @ W) - cp.trace(F @ np.linalg.inv(np.eye(m*T) + H.T @ np.eye(n*(T+1)) @ H)))

# ---------------------
# Constraints
# ---------------------
constraints = []

# Wasserstein constraints for X0
constraints += [cp.trace(X0 + X0_hat - 2*Ex0) <= rho_x0**2]
constraints += [
    cp.bmat([
        [X0_hat**0.5 @ X0 @ X0_hat**0.5, Ex0],
        [Ex0, np.eye(n)]
    ]) >> 0
]

# Wasserstein constraints for W_t
for t in range(T):
    constraints += [cp.trace(W_list[t+1] + W_hat_list[t] - 2*Ew_list[t]) <= rho_w**2]
    constraints += [
        cp.bmat([
            [W_hat_list[t]**0.5 @ W_list[t+1] @ W_hat_list[t]**0.5, Ew_list[t]],
            [Ew_list[t], np.eye(n)]
        ]) >> 0
    ]

# Wasserstein constraints for V_t
for t in range(T):
    constraints += [cp.trace(V_list[t] + V_hat_list[t] - 2*Ev_list[t]) <= rho_v**2]
    constraints += [
        cp.bmat([
            [V_hat_list[t]**0.5 @ V_list[t] @ V_hat_list[t]**0.5, Ev_list[t]],
            [Ev_list[t], np.eye(p)]
        ]) >> 0
    ]

# Big semidefinite constraint
constraints += [
    cp.bmat([
        [F, H.T @  G @ W @ D.T + M/2],
        [(H.T @ G @ W @ D.T + M/2).T, D @ W @ D.T + V]
    ]) >> 0
]

# ---------------------
# Solve the problem
# ---------------------
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK)

# ---------------------
# Print results
# ---------------------
print("\nOptimal value:", problem.value)
print("Optimal X0:\n", X0.value)
print("First W0:\n", W_list[0].value)
print("First V0:\n", V_list[0].value)
