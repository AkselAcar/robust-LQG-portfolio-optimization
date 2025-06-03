# import matplotlib.pyplot as plt
# import numpy as np
# from simulate_lqg import simulate_lqg, simulate_robust_lqg

# def simulate_and_plot_average_trajectories(T, A, B, C, mu, K_list, L_list, K_list_rob, L_list_rob, X0_robust, W_test, V_test, 
#                                            initial_point, N_runs=10
# ):
#     n = A.shape[0]
#     x_nom_all = np.zeros((N_runs, T, n))
#     x_rob_all = np.zeros((N_runs, T, n))

#     for i in range(N_runs):
#         x0 = np.random.multivariate_normal(initial_point.flatten(), X0_robust).reshape(-1, 1) # to get back to shape (n, 1)
#         x_nom, _ = simulate_lqg(T, A, B, C, mu, K_list, L_list, x0, W_test, V_test)
#         x_rob, _ = simulate_robust_lqg(T, A, B, C, mu, K_list_rob, L_list_rob, x0, W_test, V_test)
#         x_nom_all[i] = x_nom
#         x_rob_all[i] = x_rob

#     x_nom_mean = x_nom_all.mean(axis=0)  # axis = 0 for averaging over runs instead of time
#     x_rob_mean = x_rob_all.mean(axis=0)
#     x_nom_std = x_nom_all.std(axis=0)
#     x_rob_std = x_rob_all.std(axis=0)

#     time = np.arange(T)

#     # --- Plot 1: Nominal Controller ---
#     plt.figure(figsize=(8, 4))
#     for dim in range(n):
#         plt.plot(time, x_nom_mean[:, dim], label=f"State {dim+1}")
#         plt.fill_between(time,
#                          x_nom_mean[:, dim] - x_nom_std[:, dim],
#                          x_nom_mean[:, dim] + x_nom_std[:, dim],
#                          alpha=0.2) 
        
#     plt.axhline(mu.mean(), color='red', linestyle='--', linewidth=1.5, label='Target $\\mu$')
#     plt.title("Nominal LQG: Average State Trajectories")
#     plt.xlabel("Time step")
#     plt.ylabel("State value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # --- Plot 2: Robust Controller ---
#     plt.figure(figsize=(8, 4))
#     for dim in range(n):
#         plt.plot(time, x_rob_mean[:, dim], label=f"State {dim+1}")
#         plt.fill_between(time,
#                          x_rob_mean[:, dim] - x_rob_std[:, dim],
#                          x_rob_mean[:, dim] + x_rob_std[:, dim],
#                          alpha=0.2) # alpha for opacity 
#     plt.title("Robust LQG: Average State Trajectories")
#     plt.xlabel("Time step")
#     plt.ylabel("State value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # --- Plot 3: Both Controllers Superimposed ---
#     plt.figure(figsize=(8, 4))
#     for dim in range(n):
#         plt.plot(time, x_nom_mean[:, dim], label=f"Nominal - State {dim+1}", linestyle='--')
#         plt.fill_between(time,
#                          x_nom_mean[:, dim] - x_nom_std[:, dim],
#                          x_nom_mean[:, dim] + x_nom_std[:, dim],
#                          alpha=0.15)
#         plt.plot(time, x_rob_mean[:, dim], label=f"Robust - State {dim+1}")
#         plt.fill_between(time,
#                          x_rob_mean[:, dim] - x_rob_std[:, dim],
#                          x_rob_mean[:, dim] + x_rob_std[:, dim],
#                          alpha=0.15)
#     plt.title("Comparison of Nominal and Robust LQG")
#     plt.xlabel("Time step")
#     plt.ylabel("State value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     return x_nom_mean, x_rob_mean


# import matplotlib.pyplot as plt
# import numpy as np
# from simulate_lqg import simulate_lqg, simulate_robust_lqg

# def simulate_and_plot_average_trajectories(
#     T, A, B, C, mu, 
#     K_list, L_list, 
#     K_list_rob, L_list_rob, 
#     X0_robust, W_test, V_test, 
#     initial_point, N_runs=10
# ):
#     n = A.shape[0]
#     x_nom_all = np.zeros((N_runs, T, n))
#     x_rob_all = np.zeros((N_runs, T, n))

#     for i in range(N_runs):
#         x0 = np.random.multivariate_normal(initial_point.flatten(), X0_robust).reshape(-1, 1)
#         x_nom, _ = simulate_lqg(T, A, B, C, mu, K_list, L_list, x0, W_test, V_test)
#         x_rob, _ = simulate_robust_lqg(T, A, B, C, mu, K_list_rob, L_list_rob, x0, W_test, V_test)
#         x_nom_all[i] = x_nom
#         x_rob_all[i] = x_rob

#     x_nom_mean = x_nom_all.mean(axis=0)
#     x_rob_mean = x_rob_all.mean(axis=0)
#     x_nom_std = x_nom_all.std(axis=0)
#     x_rob_std = x_rob_all.std(axis=0)

#     time = np.arange(T)

#     # --- Plot Nominal Only ---
#     plt.figure(figsize=(8, 4))
#     for dim in range(n):
#         plt.plot(time, x_nom_mean[:, dim], label=f"State {dim+1}")
#         plt.fill_between(time, 
#                          x_nom_mean[:, dim] - x_nom_std[:, dim], 
#                          x_nom_mean[:, dim] + x_nom_std[:, dim], 
#                          alpha=0.2)
#         plt.axhline(y=mu[dim, 0], color='gray', linestyle='--', linewidth=1.2,
#                     label='Target $\\mu$' if dim == 0 else None)
#     plt.title("Nominal LQG: Average State Trajectories")
#     plt.xlabel("Time step")
#     plt.ylabel("State value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # --- Plot Robust Only ---
#     plt.figure(figsize=(8, 4))
#     for dim in range(n):
#         plt.plot(time, x_rob_mean[:, dim], label=f"State {dim+1}")
#         plt.fill_between(time, 
#                          x_rob_mean[:, dim] - x_rob_std[:, dim], 
#                          x_rob_mean[:, dim] + x_rob_std[:, dim], 
#                          alpha=0.2)
#         plt.axhline(y=mu[dim, 0], color='gray', linestyle='--', linewidth=1.2,
#                     label='Target $\\mu$' if dim == 0 else None)
#     plt.title("Robust LQG: Average State Trajectories")
#     plt.xlabel("Time step")
#     plt.ylabel("State value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#      # --- Separate Plots: Comparison Per State (Nominal vs Robust) ---
#     for dim in range(n):
#         plt.figure(figsize=(8, 4))
#         plt.plot(time, x_nom_mean[:, dim], label="Nominal", linestyle='--', color='tab:blue')
#         plt.fill_between(time,
#                          x_nom_mean[:, dim] - x_nom_std[:, dim],
#                          x_nom_mean[:, dim] + x_nom_std[:, dim],
#                          alpha=0.2, color='tab:blue')

#         plt.plot(time, x_rob_mean[:, dim], label="Robust", color='tab:orange')
#         plt.fill_between(time,
#                          x_rob_mean[:, dim] - x_rob_std[:, dim],
#                          x_rob_mean[:, dim] + x_rob_std[:, dim],
#                          alpha=0.2, color='tab:orange')

#         plt.axhline(y=mu[dim, 0], color='gray', linestyle='--', linewidth=1.5, label='Target $\\mu$')

#         plt.title(f"State {dim + 1}: Nominal vs Robust LQG")
#         plt.xlabel("Time step")
#         plt.ylabel("State value")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()


#     return x_nom_mean, x_rob_mean


import matplotlib.pyplot as plt
import numpy as np
from simulate_lqg import simulate_lqg, simulate_robust_lqg

def simulate_and_plot_average_trajectories(
    T, A, B, C, mu, 
    K_list, L_list, 
    K_list_rob, L_list_rob, 
    X0_robust, W_test, V_test, 
    initial_point, N_runs=10
):
    n = A.shape[0]
    x_nom_all = np.zeros((N_runs, T, n))
    x_rob_all = np.zeros((N_runs, T, n))

    for i in range(N_runs):
        x0 = np.random.multivariate_normal(initial_point.flatten(), X0_robust).reshape(-1, 1)
        x_nom, _ = simulate_lqg(T, A, B, C, mu, K_list, L_list, x0, W_test, V_test)
        x_rob, _ = simulate_robust_lqg(T, A, B, C, mu, K_list_rob, L_list_rob, x0, W_test, V_test)
        x_nom_all[i] = x_nom
        x_rob_all[i] = x_rob

    x_nom_mean = x_nom_all.mean(axis=0)
    x_rob_mean = x_rob_all.mean(axis=0)
    x_nom_std = x_nom_all.std(axis=0)
    x_rob_std = x_rob_all.std(axis=0)

    time = np.arange(T)

    # --- Create comparative plots for each state dimension ---
    plt.figure(figsize=(12, 5))
    
    # First state dimension (index 0)
    plt.subplot(1, 2, 1)
    plt.plot(time, x_nom_mean[:, 0], 'b--', label="Nominal")
    plt.fill_between(time, 
                     x_nom_mean[:, 0] - x_nom_std[:, 0], 
                     x_nom_mean[:, 0] + x_nom_std[:, 0], 
                     alpha=0.2, color='b')
    
    plt.plot(time, x_rob_mean[:, 0], 'r-', label="Robust")
    plt.fill_between(time, 
                     x_rob_mean[:, 0] - x_rob_std[:, 0], 
                     x_rob_mean[:, 0] + x_rob_std[:, 0], 
                     alpha=0.2, color='r')
    
    plt.axhline(y=mu[0, 0], color='k', linestyle='--', label='Target')
    plt.title(f"State 1 Comparison")
    plt.xlabel("Time step")
    plt.ylabel("State value")
    plt.legend()
    plt.grid(True)

    # Second state dimension (index 1)
    plt.subplot(1, 2, 2)
    plt.plot(time, x_nom_mean[:, 1], 'b--', label="Nominal")
    plt.fill_between(time, 
                     x_nom_mean[:, 1] - x_nom_std[:, 1], 
                     x_nom_mean[:, 1] + x_nom_std[:, 1], 
                     alpha=0.2, color='b')
    
    plt.plot(time, x_rob_mean[:, 1], 'r-', label="Robust")
    plt.fill_between(time, 
                     x_rob_mean[:, 1] - x_rob_std[:, 1], 
                     x_rob_mean[:, 1] + x_rob_std[:, 1], 
                     alpha=0.2, color='r')
    
    plt.axhline(y=mu[1, 0], color='k', linestyle='--', label='Target')
    plt.title(f"State 2 Comparison")
    plt.xlabel("Time step")
    plt.ylabel("State value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return x_nom_mean, x_rob_mean
