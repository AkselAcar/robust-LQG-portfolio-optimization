import numpy as np

def simulate_lqg(N, a, b, c, mu, K, L,sigma_x0, sigma_w, sigma_v, dt, x0):
    x, z, y, z_hat, u = [np.zeros(N) for _ in range(5)]
    x[0] = x0
    z[0] = mu - x0
    z_hat[0] = z[0] + np.random.normal(0, sigma_x0) # sigama_x0 here ???

    for k in range(1, N):
        u[k] = -K * z_hat[k-1]
        w_k = np.random.normal(0, sigma_w * np.sqrt(dt))
        z[k] = a * z[k-1] + b * u[k] + w_k
        x[k] = mu - z[k]
        v_k = np.random.normal(0, sigma_v)
        y[k] = c * z[k] + v_k
        z_hat[k] = a * z_hat[k-1] + b * u[k] + L * (y[k] - c * (a * z_hat[k-1] + b * u[k]))
    
    return x, z, z_hat, u, y

def simulate_robust_lqg(N, a, b, c, mu, K, L_rob, Qw_rob, Rv_rob, dt, x0):
    x, z, z_hat, u = [np.zeros(N) for _ in range(4)]
    x[0] = x0
    z[0] = mu - x0
    z_hat[0] = z[0] + np.random.normal(0, np.sqrt(Rv_rob))

    for k in range(1, N):
        u[k] = -K * z_hat[k-1]
        w_k = np.random.normal(0, np.sqrt(Qw_rob))
        z[k] = a * z[k-1] + b * u[k] + w_k
        x[k] = mu - z[k]
        v_k = np.random.normal(0, np.sqrt(Rv_rob))
        y_k = c * z[k] + v_k
        z_hat[k] = a * z_hat[k-1] + b * u[k] + L_rob * (y_k - c * (a * z_hat[k-1] + b * u[k]))
    
    return x, z, z_hat, u


import numpy as np

def simulate_lqg(N, A, B, C, mu, K, L, Qw, Rv, x0):
    n = A.shape[0]  # state dimension
    m = B.shape[1]  # input dimension
    p = C.shape[0]  # output dimension

    x = np.zeros((N, n))
    z = np.zeros((N, n))
    y = np.zeros((N, p))
    z_hat = np.zeros((N, n))
    u = np.zeros((N, m))

    x[0] = x0
    z[0] = mu - x0
    z_hat[0] = z[0] + np.random.multivariate_normal(np.zeros(p), Rv)

    for k in range(1, N):
        u[k] = -K @ z_hat[k - 1]
        w_k = np.random.multivariate_normal(np.zeros(n), Qw)
        z[k] = A @ z[k - 1] + B @ u[k] + w_k
        x[k] = mu - z[k]
        v_k = np.random.multivariate_normal(np.zeros(p), Rv)
        y[k] = C @ z[k] + v_k
        z_pred = A @ z_hat[k - 1] + B @ u[k]
        z_hat[k] = z_pred + L @ (y[k] - C @ z_pred)

    return x, z, z_hat, u, y


def simulate_robust_lqg(N, A, B, C, mu, K, L_rob, Qw_rob, Rv_rob, x0):
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    x = np.zeros((N, n))
    z = np.zeros((N, n))
    z_hat = np.zeros((N, n))
    u = np.zeros((N, m))

    x[0] = x0
    z[0] = mu - x0
    z_hat[0] = z[0] + np.random.multivariate_normal(np.zeros(p), Rv_rob)

    for k in range(1, N):
        u[k] = -K @ z_hat[k - 1]
        w_k = np.random.multivariate_normal(np.zeros(n), Qw_rob)
        z[k] = A @ z[k - 1] + B @ u[k] + w_k
        x[k] = mu - z[k]
        v_k = np.random.multivariate_normal(np.zeros(p), Rv_rob)
        y_k = C @ z[k] + v_k
        z_pred = A @ z_hat[k - 1] + B @ u[k]
        z_hat[k] = z_pred + L_rob @ (y_k - C @ z_pred)

    return x, z, z_hat, u
