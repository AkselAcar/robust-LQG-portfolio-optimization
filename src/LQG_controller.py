# def compute_lqr_gain(a, b, Q, R, iterations=1000):
#     P = Q
#     for _ in range(iterations):
#         P = Q + a**2 * P - (a * b * P)**2 / (R + b**2 * P)
#     K = (a * b * P) / (R + b**2 * P)
#     return K

# def compute_kalman_gain(a, c, Qw, Rv, Sigma_init, iterations=1000):
#     Sigma = Sigma_init
#     for _ in range(iterations):
#         Sigma = a**2 * Sigma + Qw - (a * Sigma * c)**2 / (c**2 * Sigma + Rv)
#     L = a * Sigma * c / (c**2 * Sigma + Rv)
#     return L

import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are

def compute_lqr_gain(A, B, Q, R):
    """
    Computes the LQR gain matrix using the discrete-time algebraic Riccati equation.
    """
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P

def compute_kalman_gain(A, C, Qw, Rv, Sigma_init=None, iterations=1000, tol=1e-8):
    """
    Computes the steady-state Kalman gain L for the system:

    x_{k+1} = A x_k + w_k
    y_k = C x_k + v_k

    Returns:
        L: Kalman gain (n x p)
        Sigma: Steady-state error covariance
    """
    n = A.shape[0]
    Sigma = Sigma_init if Sigma_init is not None else 1e-3 * np.eye(n)

    for _ in range(iterations):
        Sigma_pred = A @ Sigma @ A.T + Qw
        S = C @ Sigma_pred @ C.T + Rv
        L = Sigma_pred @ C.T @ np.linalg.inv(S)
        Sigma_new = (np.eye(n) - L @ C) @ Sigma_pred
        if np.linalg.norm(Sigma_new - Sigma) < tol:
            break
        Sigma = Sigma_new

    return L, Sigma
