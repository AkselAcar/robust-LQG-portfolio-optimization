import numpy as np

def simulate_lqg(T, A, B, C, mu, K_list, L_list, x0, W_list, V_list, Q, R): 
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    x = np.zeros((T + 1, n))
    z = np.zeros((T + 1, n))
    z_hat = np.zeros((T + 1, n))
    y = np.zeros((T, p))  # Measurements at each time step
    u = np.zeros((T, m))
    
    # Initialize cost accumulator
    cost_vector = np.zeros(T)
    
    x[0] = x0.flatten()
    z[0] = (mu - x0).flatten()
    z_hat[0] = z[0] + np.random.multivariate_normal(np.zeros(p), V_list[0])
    
    # # Initial state cost - z[0]^T Q z[0]
    # cost_vector[0] += z[0].T @ Q @ z[0]

    # for k in range(1, T):
    #     K = K_list[k - 1]
    #     L = L_list[k]
    #     w_k = np.random.multivariate_normal(np.zeros(n), W_list[k])
    #     v_k = np.random.multivariate_normal(np.zeros(p), V_list[k])

    #     u[k] = -K @ z_hat[k - 1]  # ??? change it to same k 
    #     z[k] = A @ z[k - 1] + B @ u[k] + w_k
    #     x[k] = mu.flatten() - z[k]
    #     y[k] = C @ z[k] + v_k
    #     z_pred = A @ z_hat[k - 1] + B @ u[k]
    #     z_hat[k] = z_pred + L @ (y[k] - C @ z_pred)
        
    #     # cost at each timestep - z[k]^T Q z[k] + u[k]^T R u[k]
    #     cost_vector[k] += z[k].T @ Q @ z[k] + u[k].T @ R @ u[k]
    
    for k in range(T):
        K = K_list[k]
        L = L_list[k]
        w_k = np.random.multivariate_normal(np.zeros(n), W_list[k])
        v_k = np.random.multivariate_normal(np.zeros(p), V_list[k])

        u[k] = -K @ z_hat[k]  
        z[k + 1] = A @ z[k] + B @ u[k] + w_k
        x[k + 1] = mu.flatten() - z[k + 1]
        y_k = C @ z[k + 1] + v_k
        z_pred = A @ z_hat[k] + B @ u[k]
        z_hat[k + 1] = z_pred + L @ (y_k - C @ z_pred)
        
        # cost at each timestep - z[k]^T Q z[k] + u[k]^T R u[k]
        cost_vector[k] += z[k].T @ Q @ z[k] + u[k].T @ R @ u[k]

    # Terminal state cost - z[-1]^T Q z[-1]
    cost_vector[-1] += z[-1].T @ Q @ z[-1]

    # Total cost is the sum of all costs at each timestep
    total_cost = np.sum(cost_vector)
    
    return x, u, total_cost, cost_vector

def simulate_robust_lqg(T, A, B, C, mu, K_list, L_list, x0_rob, W_list, V_list, Q, R):  
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    x = np.zeros((T + 1, n))
    z = np.zeros((T + 1, n))
    z_hat = np.zeros((T + 1, n))
    y = np.zeros((T, p))  # Measurements at each time step
    u = np.zeros((T, m))
    
    # Initialize cost accumulator
    cost_vector = np.zeros(T)
    
    x[0] = x0_rob.flatten()
    z[0] = (mu - x0_rob).flatten()
    z_hat[0] = z[0] + np.random.multivariate_normal(np.zeros(p), V_list[0])



    # # Initial state cost - z[0]^T Q z[0]
    # cost_vector[0] += z[0].T @ Q @ z[0]

    # for k in range(1, T):
    #     K = K_list[k - 1]
    #     L = L_list[k]
    #     w_k = np.random.multivariate_normal(np.zeros(n), W_list[k])
    #     v_k = np.random.multivariate_normal(np.zeros(p), V_list[k])

    #     u[k] = -K @ z_hat[k - 1]  # ??? change it to same k 
    #     z[k] = A @ z[k - 1] + B @ u[k] + w_k
    #     x[k] = mu.flatten() - z[k]
    #     y[k] = C @ z[k] + v_k
    #     z_pred = A @ z_hat[k - 1] + B @ u[k]
    #     z_hat[k] = z_pred + L @ (y[k] - C @ z_pred)
        
    #     # cost at each timestep - z[k]^T Q z[k] + u[k]^T R u[k]
    #     cost_vector[k] += z[k].T @ Q @ z[k] + u[k].T @ R @ u[k]
    
    
    for k in range(T):
        K = K_list[k]
        L = L_list[k]
        w_k = np.random.multivariate_normal(np.zeros(n), W_list[k])
        v_k = np.random.multivariate_normal(np.zeros(p), V_list[k])

        u[k] = -K @ z_hat[k]
        z[k + 1] = A @ z[k] + B @ u[k] + w_k
        x[k + 1] = mu.flatten() - z[k + 1]
        y_k = C @ z[k +1] + v_k
        z_pred = A @ z_hat[k] + B @ u[k]
        z_hat[k + 1] = z_pred + L @ (y_k - C @ z_pred).flatten()
        
        # Cost at each timestep
        cost_vector[k] += z[k].T @ Q @ z[k] + u[k].T @ R @ u[k] # cost at each step, 
    
    # Terminal state cost
    cost_vector[-1] += z[-1].T @ Q @ z[-1]

    # Total cost is the sum of all costs at each timestep
    total_cost = np.sum(cost_vector)
    
    return x, u, total_cost, cost_vector