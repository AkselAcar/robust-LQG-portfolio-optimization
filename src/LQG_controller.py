import numpy as np
from scipy.linalg import solve_discrete_are

def compute_finite_horizon_lqr_gains(A, B, Q, R, T):
    """
    Computes a list of finite-horizon LQR gains using backward Riccati recursion.

    Returns:
        K_list: List of LQR gains [K_0, ..., K_{T-1}]
    """
    n = A.shape[0]
    m = B.shape[1]

    P = Q  # Terminal cost
    P_list = [None] * (T + 1)
    K_list = [None] * T
    P_list[T] = P

    for t in reversed(range(T)):
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        K_list[t] = K
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        P_list[t] = P

    return K_list

    
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