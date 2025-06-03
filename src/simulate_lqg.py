# import numpy as np

# def simulate_lqg(N, A, B, C, mu, K, L, Qw, Rv, x0):
#     n = A.shape[0]  # state dimension
#     m = B.shape[1]  # input dimension
#     p = C.shape[0]  # output dimension

#     x = np.zeros((N, n))
#     z = np.zeros((N, n))
#     y = np.zeros((N, p))
#     z_hat = np.zeros((N, n))
#     u = np.zeros((N, m))

#     x[0] = x0
#     z[0] = mu - x0
#     z_hat[0] = z[0] + np.random.multivariate_normal(np.zeros(p), Rv)

#     for k in range(1, N):
#         u[k] = -K @ z_hat[k - 1]
#         w_k = np.random.multivariate_normal(np.zeros(n), Qw)
#         z[k] = A @ z[k - 1] + B @ u[k] + w_k
#         x[k] = mu - z[k]
#         v_k = np.random.multivariate_normal(np.zeros(p), Rv)
#         y[k] = C @ z[k] + v_k
#         z_pred = A @ z_hat[k - 1] + B @ u[k]
#         z_hat[k] = z_pred + L @ (y[k] - C @ z_pred)

#     return x, z, z_hat, u, y


# def simulate_robust_lqg(N, A, B, C, mu, K, L_rob, Qw_rob, Rv_rob, x0):
#     n = A.shape[0]
#     m = B.shape[1]
#     p = C.shape[0]

#     x = np.zeros((N, n))
#     z = np.zeros((N, n))
#     z_hat = np.zeros((N, n))
#     u = np.zeros((N, m))

#     x[0] = x0
#     z[0] = mu - x0
#     z_hat[0] = z[0] + np.random.multivariate_normal(np.zeros(p), Rv_rob)

#     for k in range(1, N):
#         u[k] = -K @ z_hat[k - 1]
#         w_k = np.random.multivariate_normal(np.zeros(n), Qw_rob)
#         z[k] = A @ z[k - 1] + B @ u[k] + w_k
#         x[k] = mu - z[k]
#         v_k = np.random.multivariate_normal(np.zeros(p), Rv_rob)
#         y_k = C @ z[k] + v_k
#         z_pred = A @ z_hat[k - 1] + B @ u[k]
#         z_hat[k] = z_pred + L_rob @ (y_k - C @ z_pred)

#     return x, z, z_hat, u


import numpy as np

def simulate_lqg(T, A, B, C, mu, K_list, L_list,x0, Qw, Rv):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    x = np.zeros((T, n))
    z = np.zeros((T, n))
    y = np.zeros((T, p))
    z_hat = np.zeros((T, n))
    u = np.zeros((T, m))

    x[0] = x0.flatten()
    z[0] = (mu - x0).flatten()
    z_hat[0] = z[0] + np.random.multivariate_normal(np.zeros(p), Rv)

    for k in range(1, T):
        K = K_list[k - 1]  # Time-varying LQR gain (already reversed in the LQR computation)
        L = L_list[k]      # Time-varying Kalman gain

        u[k] = -K @ z_hat[k - 1] # We took the control action based on the previous estimated states as K is K[k-1]
        w_k = np.random.multivariate_normal(np.zeros(n), Qw)
        z[k] = A @ z[k - 1] + B @ u[k] + w_k
        x[k] = mu.flatten() - z[k]
        v_k = np.random.multivariate_normal(np.zeros(p), Rv)
        y[k] = C @ z[k] + v_k
        z_pred = A @ z_hat[k - 1] + B @ u[k]
        z_hat[k] = z_pred + L @ (y[k] - C @ z_pred)

    return x, u 


def simulate_robust_lqg(T, A, B, C, mu, K_list, L_list, x0, Qw_rob, Rv_rob,):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    x = np.zeros((T, n))
    z = np.zeros((T, n))
    z_hat = np.zeros((T, n))
    u = np.zeros((T, m))

    x[0] = x0.flatten()
    z[0] = (mu - x0).flatten()
    z_hat[0] = z[0] + np.random.multivariate_normal(np.zeros(p), Rv_rob)

    for k in range(1, T):
        K = K_list[k - 1]
        L = L_list[k]

        u[k] = -K @ z_hat[k - 1]
        w_k = np.random.multivariate_normal(np.zeros(n), Qw_rob)
        z[k] = A @ z[k - 1] + B @ u[k] + w_k
        x[k] = mu.flatten() - z[k]
        v_k = np.random.multivariate_normal(np.zeros(p), Rv_rob)
        y_k = C @ z[k] + v_k
        z_pred = A @ z_hat[k - 1] + B @ u[k]
        z_hat[k] = z_pred + L @ (y_k - C @ z_pred)

    return x, u

