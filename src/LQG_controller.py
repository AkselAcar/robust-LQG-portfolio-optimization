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