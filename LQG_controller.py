import numpy as np

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


def finite_horizon_kalman_gains(A, C, W_list, V_list, Sigma0, T):
    """
    Computes finite-horizon Kalman gains with time-varying covariances.
    
    Returns:
        L_list: Kalman gains for each time step
    """
    n = A.shape[0]
    p = C.shape[0]
    Sigma = Sigma0.copy()
    
    L_list = []

    for k in range(T):
        # Get current time-step covariances
        W = W_list[k] if k < len(W_list) else W_list[-1] # Not necessary but for safety, if W_list is shorter than T use the last element
        V = V_list[k] if k < len(V_list) else V_list[-1]
        
        # Compute Kalman gain
        S = C @ Sigma @ C.T + V
        L = A @ Sigma @ C.T @ np.linalg.inv(S)
        L_list.append(L)
        
        # Update covariance
        Sigma = A @ Sigma @ A.T + W - A @ Sigma @ C.T @ np.linalg.inv(S) @ C @ Sigma @ A.T
        
    return L_list


