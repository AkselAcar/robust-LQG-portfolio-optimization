import cvxpy as cp
import numpy as np
import scipy as sp

def solve_sdp(
    A,
    B,
    C,
    sigma_x0,
    sigma_w,
    sigma_v,
    dt,
    T,
    rho_x0=1.0,
    rho_w=1.0,
    rho_v=1.0,
    solver=cp.MOSEK
):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    # Variance matrices
    X0_hat = sigma_x0**2 * np.eye(n)
    W_hat_list = [sigma_w**2 * dt * np.eye(n) for _ in range(T)]
    V_hat_list = [sigma_v**2 * np.eye(p) for _ in range(T)]

    # --- Variables ---
    X0 = cp.Variable((n, n), PSD=True)
    W_list = [cp.Variable((n, n), PSD=True) for _ in range(T)]
    V_list = [cp.Variable((p, p), PSD=True) for _ in range(T)]
    F = cp.Variable((T * m, T * m), PSD=True)

    Ex0 = cp.Variable((n, n), PSD=True)
    Ew_list = [cp.Variable((n, n), PSD=True) for _ in range(T)]
    Ev_list = [cp.Variable((p, p), PSD=True) for _ in range(T)]

    # --- G Matrix ---
    G = np.zeros((n * (T + 1), n * (T + 1)))
    for t in range(T + 1):
        for s in range(t):
            A_prod = np.linalg.matrix_power(A, t - s)
            G[n*t:n*(t+1), n*s:n*(s+1)] = A_prod
        G[n*t:n*(t+1), n*t:n*(t+1)] = np.eye(n)

    # --- H Matrix ---
    H = np.zeros((n * (T + 1), m * T))
    for t in range(1, T + 1):
        for s in range(t):
            A_prod = np.linalg.matrix_power(A, t - s - 1)
            H[n*t:n*(t+1), m*s:m*(s+1)] = A_prod @ B

    # --- C_big and D ---
    C_blocks = []
    for t in range(T):
        row = [np.zeros((p, n)) for _ in range(T + 1)]
        row[t] = C
        C_blocks.append(row)
    C_big = np.block(C_blocks)
    D = C_big @ G

    # --- W and V Block Matrices ---
    W_blocks = [X0] + W_list
    W = cp.bmat([
        [W_blocks[i] if i == j else np.zeros((n, n)) for j in range(T + 1)]
        for i in range(T + 1)
    ])
    V = cp.bmat([
        [V_list[i] if i == j else np.zeros((p, p)) for j in range(T)]
        for i in range(T)
    ])

    # --- M Matrix ---
    M = cp.bmat([
        [cp.Variable((m, p)) if i < j else np.zeros((m, p)) for j in range(T)]
        for i in range(T)
    ])

    Q_mat = np.eye(n * (T + 1))
    R_mat = np.eye(p * T)

    # --- Objective ---
    objective = cp.Maximize(
        cp.trace(G.T @ Q_mat @ G @ W) - cp.trace(F @ np.linalg.inv(R_mat + H.T @ Q_mat @ H))
    )

    # --- Constraints ---
    constraints = []

    # Wasserstein constraints for X0
    sqrt_X0_hat = sp.linalg.sqrtm(X0_hat)
    min_eig_X0 = sp.linalg.eigvalsh(X0_hat)[0]
    constraints += [X0 - min_eig_X0 * np.eye(n) >> 0]
    constraints += [cp.trace(X0 + X0_hat - 2 * Ex0) <= rho_x0**2]
    constraints += [cp.bmat([[sqrt_X0_hat @ X0 @ sqrt_X0_hat, Ex0],
                             [Ex0, np.eye(n)]]) >> 0]

    # Wasserstein constraints for W_t
    for t in range(T):
        sqrt_W_hat = sp.linalg.sqrtm(W_hat_list[t])
        min_eig_W = sp.linalg.eigvalsh(W_hat_list[t])[0]
        constraints += [W_list[t] - min_eig_W * np.eye(n) >> 0]
        constraints += [cp.trace(W_list[t] + W_hat_list[t] - 2 * Ew_list[t]) <= rho_w**2]
        constraints += [cp.bmat([[sqrt_W_hat @ W_list[t] @ sqrt_W_hat, Ew_list[t]],
                                 [Ew_list[t], np.eye(n)]]) >> 0]

    # Wasserstein constraints for V_t
    for t in range(T):
        sqrt_V_hat = sp.linalg.sqrtm(V_hat_list[t])
        min_eig_V = sp.linalg.eigvalsh(V_hat_list[t])[0]
        constraints += [V_list[t] - min_eig_V * np.eye(p) >> 0]
        constraints += [cp.trace(V_list[t] + V_hat_list[t] - 2 * Ev_list[t]) <= rho_v**2]
        constraints += [cp.bmat([[sqrt_V_hat @ V_list[t] @ sqrt_V_hat, Ev_list[t]],
                                 [Ev_list[t], np.eye(p)]]) >> 0]

    # Big LMI constraint
    constraints += [
        cp.bmat([[F, H.T @ Q_mat @ G @ W @ D.T + M / 2],
                 [(H.T @ Q_mat @ G @ W @ D.T + M / 2).T, D @ W @ D.T + V]]) >> 0
    ]

    # Solve problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    # Return first (initial) robustified covariances
    return {
        "status": problem.status,
        "optimal_value": problem.value,
        "X0": X0.value,
        "W0": W_list[0].value,
        "V0": V_list[0].value
    }
