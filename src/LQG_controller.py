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
from scipy.linalg import solve_discrete_are

def compute_lqr_gain(A, B, Q, R): # Change into finite-horizon LQR
    """
    Computes the LQR gain matrix using the discrete-time algebraic Riccati equation.
    """
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P
    
def finite_horizon_kalman_gains(A, C, W, V, Sigma0, T):
    """
    Computes finite-horizon Kalman gains using the DRE as shown in the provided slide.

    """
    # n = A.shape[0]
    Sigma = Sigma0.copy()

    L_list = []
    Sigma_list = [Sigma0]

    for _ in range(T):
        # Compute Kalman gain from DRE prediction
        S = C @ Sigma @ C.T + V
        L = A @ Sigma @ C.T @ np.linalg.inv(S)
        L_list.append(L)

        # DRE Update for Σ_{k+1|k}
        Sigma = A @ Sigma @ A.T + W - A @ Sigma @ C.T @ np.linalg.inv(S) @ C @ Sigma @ A.T
        Sigma_list.append(Sigma)

    return L_list, Sigma_list