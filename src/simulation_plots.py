# import matplotlib.pyplot as plt
# import numpy as np
# from simulate_lqg import simulate_lqg, simulate_robust_lqg

# def simulate_and_plot_average_trajectories(
#     T, A, B, C, mu, 
#     K_list, L_list, 
#     K_list_rob, L_list_rob, 
#     X0_robust, W_test, V_test, 
#     initial_point, N_runs=10,
#     Q=None, R=None 
# ):
#     n = A.shape[0]
#     # Arrays to store cost trajectories
#     cost_nom_all = np.zeros((N_runs, T))
#     cost_rob_all = np.zeros((N_runs, T))
    
#     # Arrays to store total costs per run
#     total_cost_nom = np.zeros(N_runs)
#     total_cost_rob = np.zeros(N_runs)
    
#     # Arrays for states
#     x_nom_all = np.zeros((N_runs, T, n))
#     x_rob_all = np.zeros((N_runs, T, n))

#     for i in range(N_runs):
#         x0 = np.random.multivariate_normal(initial_point.flatten(), X0_robust).reshape(-1, 1)
        
#         # Simulate
#         x_nom, u_nom, total_cost_nom_one_run, cost_nom_vector = simulate_lqg(T, A, B, C, mu, K_list, L_list, x0, W_test, V_test, Q, R)
#         x_rob, u_rob, total_cost_rob_one_run, cost_rob_vector = simulate_robust_lqg(T, A, B, C, mu, K_list_rob, L_list_rob, x0, W_test, V_test, Q, R)
        
#         # Store states
#         x_nom_all[i] = x_nom
#         x_rob_all[i] = x_rob
        
#         # Store total cost trajectories of each run
#         total_cost_nom[i] = total_cost_nom_one_run # vector of length N_runs
#         total_cost_rob[i] = total_cost_rob_one_run
#         # Store total costs per time step per run 
#         cost_nom_all[i] = cost_nom_vector # a vector of costs at each time step, basically a vector of length N_runs of vectors of length T
#         cost_rob_all[i] = cost_rob_vector

#     # Compute averages for states
#     x_nom_mean = x_nom_all.mean(axis=0)
#     x_rob_mean = x_rob_all.mean(axis=0)
#     x_nom_std = x_nom_all.std(axis=0)
#     x_rob_std = x_rob_all.std(axis=0)
    
#    # Per-time-step average and std (for time plots)
#     mean_cost_nom_per_timestep = cost_nom_all.mean(axis=0)  # shape (T,)
#     mean_cost_rob_per_timestep = cost_rob_all.mean(axis=0)

#     std_cost_nom_per_timestep = cost_nom_all.std(axis=0)
#     std_cost_rob_per_timestep = cost_rob_all.std(axis=0)

#     # Total cost across each run (for scatter plots and mean comparison), average of all the runs total costs
#     mean_total_cost_nom = total_cost_nom.mean()  # scalar 
#     mean_total_cost_rob = total_cost_rob.mean()

#     time = np.arange(T)

#     # --- Plot 1: Nominal Controller ---
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

#     # --- Plot 2: Robust Controller ---
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

#     # --- Plot 3: Both Controllers Superimposed ---
#     plt.figure(figsize=(8, 4))
#     for dim in range(n):
#         plt.plot(time, x_nom_mean[:, dim], linestyle='--', label=f"Nominal - State {dim+1}")
#         plt.fill_between(time,
#                          x_nom_mean[:, dim] - x_nom_std[:, dim],
#                          x_nom_mean[:, dim] + x_nom_std[:, dim],
#                          alpha=0.15)
#         plt.plot(time, x_rob_mean[:, dim], label=f"Robust - State {dim+1}")
#         plt.fill_between(time,
#                          x_rob_mean[:, dim] - x_rob_std[:, dim],
#                          x_rob_mean[:, dim] + x_rob_std[:, dim],
#                          alpha=0.15)
#         plt.axhline(y=mu[dim, 0], color='gray', linestyle='--', linewidth=1.2,
#                     label='Target $\\mu$' if dim == 0 else None)
#     plt.title("Comparison of Nominal and Robust LQG")
#     plt.xlabel("Time step")
#     plt.ylabel("State value")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # NEW: Plot 4 - Cost Trajectories Comparison ---
#     plt.figure(figsize=(10, 6))
    
#     # Instantaneous cost
#     plt.subplot(2, 1, 1)
#     plt.plot(time, cost_nom_mean, 'b-', label='Nominal Cost')
#     plt.fill_between(time, 
#                      cost_nom_mean - cost_nom_std, 
#                      cost_nom_mean + cost_nom_std, 
#                      alpha=0.2, color='blue')
#     plt.plot(time, cost_rob_mean, 'r-', label='Robust Cost')
#     plt.fill_between(time, 
#                      cost_rob_mean - cost_rob_std, 
#                      cost_rob_mean + cost_rob_std, 
#                      alpha=0.2, color='red')
#     plt.title("Instantaneous Cost Comparison")
#     plt.xlabel("Time step")
#     plt.ylabel("Cost")
#     plt.legend()
#     plt.grid(True)
    
#     # Cumulative cost
#     plt.subplot(2, 1, 2)
#     plt.plot(time, np.cumsum(cost_nom_mean), 'b-', label='Nominal Cumulative Cost')
#     plt.plot(time, np.cumsum(cost_rob_mean), 'r-', label='Robust Cumulative Cost')
#     plt.title("Cumulative Cost Comparison")
#     plt.xlabel("Time step")
#     plt.ylabel("Cumulative Cost")
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.show()


#     # --- NEW: Plot 5 - Total Cost Per Run ---
#     plt.figure(figsize=(10, 5))
#     runs = np.arange(1, N_runs + 1)
    
#     plt.scatter(runs, total_cost_nom, color='blue', label='Nominal Cost', marker='o')
#     plt.scatter(runs, total_cost_rob, color='red', label='Robust Cost', marker='x')
    
#     plt.axhline(avg_total_cost_nom, color='blue', linestyle='--', linewidth=1,
#                 label=f"Avg Nominal Cost = {avg_total_cost_nom:.1f}")
#     plt.axhline(avg_total_cost_rob, color='red', linestyle='--', linewidth=1,
#                 label=f"Avg Robust Cost = {avg_total_cost_rob:.1f}")
    
#     plt.title("Total Cost per Run")
#     plt.xlabel("Run Index")
#     plt.ylabel("Total Cost")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


#     # NEW: Print cost comparison
#     print("\n=== Cost Comparison ===")
#     print(f"Average Total Cost (Nominal): {avg_total_cost_nom:.2f}")
#     print(f"Average Total Cost (Robust): {avg_total_cost_rob:.2f}")
#     print(f"Cost Difference: {avg_total_cost_nom - avg_total_cost_rob:.2f}")

#     return x_nom_mean, x_rob_mean, avg_total_cost_nom, avg_total_cost_rob

import matplotlib.pyplot as plt
import numpy as np
from simulate_lqg import simulate_lqg, simulate_robust_lqg

def simulate_and_plot_average_trajectories(
    T, A, B, C, mu, 
    K_list, L_list, 
    K_list_rob, L_list_rob, 
    X0_robust, W_test, V_test, 
    initial_point, N_runs=10,
    Q=None, R=None 
):
    n = A.shape[0]
    # Arrays to store cost trajectories
    cost_nom_all = np.zeros((N_runs, T ))
    cost_rob_all = np.zeros((N_runs, T ))
    
    # Arrays to store total costs per run
    total_cost_nom = np.zeros(N_runs)
    total_cost_rob = np.zeros(N_runs)
    
    # Arrays for states
    x_nom_all = np.zeros((N_runs, T + 1, n))
    x_rob_all = np.zeros((N_runs, T + 1, n))

    for i in range(N_runs):
        x0 = np.random.multivariate_normal(initial_point.flatten(), X0_robust).reshape(-1, 1)
        
        # Simulate
        x_nom, u_nom, total_cost_nom_one_run, cost_nom_vector = simulate_lqg(T, A, B, C, mu, K_list, L_list, x0, W_test, V_test, Q, R)
        x_rob, u_rob, total_cost_rob_one_run, cost_rob_vector = simulate_robust_lqg(T, A, B, C, mu, K_list_rob, L_list_rob, x0, W_test, V_test, Q, R)
        
        # Store states
        x_nom_all[i] = x_nom
        x_rob_all[i] = x_rob
        
        # Store costs
        total_cost_nom[i] = total_cost_nom_one_run # vector of length N_runs
        total_cost_rob[i] = total_cost_rob_one_run
        cost_nom_all[i] = cost_nom_vector # a vector of costs at each time step, basically a vector of length N_runs of vectors of length T
        cost_rob_all[i] = cost_rob_vector

    # Compute averages for states
    x_nom_mean = x_nom_all.mean(axis=0) # Average state trajectory (T, n)
    x_rob_mean = x_rob_all.mean(axis=0)
    x_nom_std = x_nom_all.std(axis=0)
    x_rob_std = x_rob_all.std(axis=0)
    
    # Compute cost statistics
    mean_cost_nom_per_timestep = cost_nom_all.mean(axis=0)  # Average cost per timestep (T,)
    mean_cost_rob_per_timestep = cost_rob_all.mean(axis=0)
    std_cost_nom_per_timestep = cost_nom_all.std(axis=0)
    std_cost_rob_per_timestep = cost_rob_all.std(axis=0)
    
    mean_total_cost_nom = total_cost_nom.mean()  # Scalar average total cost
    mean_total_cost_rob = total_cost_rob.mean()

    time_state = np.arange(T + 1)  # For state trajectories (0 to T)
    time_cost = np.arange(T)        # For cost trajectories (0 to T-1)

    # --- Plot 1: State Trajectories Comparison ---
    plt.figure(figsize=(10, 8))
    
    # State trajectories
    plt.subplot(2, 1, 1)
    for dim in range(n):
        plt.plot(time_state, x_nom_mean[:, dim], 'b--', label=f"Nominal - State {dim+1}")
        plt.fill_between(time_state,
                         x_nom_mean[:, dim] - x_nom_std[:, dim],
                         x_nom_mean[:, dim] + x_nom_std[:, dim],
                         alpha=0.15, color='blue')
        plt.plot(time_state, x_rob_mean[:, dim], 'r-', label=f"Robust - State {dim+1}")
        plt.fill_between(time_state,
                         x_rob_mean[:, dim] - x_rob_std[:, dim],
                         x_rob_mean[:, dim] + x_rob_std[:, dim],
                         alpha=0.15, color='red')
        plt.axhline(y=mu[dim, 0], color='gray', linestyle='--', linewidth=1.2,
                    label='Target $\\mu$' if dim == 0 else None)
    plt.title("State Trajectories Comparison")
    plt.xlabel("Time step")
    plt.ylabel("State value")
    plt.legend()
    plt.grid(True)
    
    # Instantaneous cost comparison
    plt.subplot(2, 1, 2)
    plt.plot(time_cost, mean_cost_nom_per_timestep, 'b--', label='Nominal Cost')
    plt.fill_between(time_cost, 
                     mean_cost_nom_per_timestep - std_cost_nom_per_timestep, 
                     mean_cost_nom_per_timestep + std_cost_nom_per_timestep, 
                     alpha=0.2, color='blue')
    plt.plot(time_cost, mean_cost_rob_per_timestep, 'r-', label='Robust Cost')
    plt.fill_between(time_cost, 
                     mean_cost_rob_per_timestep - std_cost_rob_per_timestep, 
                     mean_cost_rob_per_timestep + std_cost_rob_per_timestep, 
                     alpha=0.2, color='red')
    plt.title("Instantaneous Cost Comparison")
    plt.xlabel("Time step")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Cumulative Costs ---
    plt.figure(figsize=(10, 5))
    plt.plot(time_cost, np.cumsum(mean_cost_nom_per_timestep), 'b--', label='Nominal Cumulative Cost')
    plt.plot(time_cost, np.cumsum(mean_cost_rob_per_timestep), 'r-', label='Robust Cumulative Cost')
    plt.title("Cumulative Cost Comparison")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Total Cost Per Run ---
    plt.figure(figsize=(10, 6))
    runs = np.arange(1, N_runs + 1)
    
    # Scatter plot of total costs
    plt.scatter(runs, total_cost_nom, color='blue', label='Nominal Cost', marker='o', s=80, alpha=0.7)
    plt.scatter(runs, total_cost_rob, color='red', label='Robust Cost', marker='x', s=80, alpha=0.7)
    
    # Average lines
    plt.axhline(mean_total_cost_nom, color='blue', linestyle='--', linewidth=2,
                label=f"Avg Nominal = {mean_total_cost_nom:.1f}")
    plt.axhline(mean_total_cost_rob, color='red', linestyle='-', linewidth=2,
                label=f"Avg Robust = {mean_total_cost_rob:.1f}")
    
    plt.title("Total Cost per Run Comparison")
    plt.xlabel("Run Index")
    plt.ylabel("Total Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print final cost comparison
    print("\n=== FINAL COST COMPARISON ===")
    print(f"Average Total Cost - Nominal: {mean_total_cost_nom:.2f}")
    print(f"Average Total Cost - Robust:  {mean_total_cost_rob:.2f}")
    print(f"Cost Difference: {mean_total_cost_nom - mean_total_cost_rob:.2f}")

    return x_nom_mean, x_rob_mean, mean_total_cost_nom, mean_total_cost_rob