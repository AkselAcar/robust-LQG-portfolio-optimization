# noise_cost_sensitivity.py
import numpy as np
import matplotlib.pyplot as plt
from simulate_lqg import simulate_lqg, simulate_robust_lqg

def analyze_noise_sensitivity(
    T, A, B, C, mu, 
    K_list, L_list, 
    K_list_rob, L_list_rob, 
    X0_hat, W_hat_list, V_hat_list,
    initial_point, 
    Q, R,
    noise_factors,
    N_runs=20
):
    """
    Analyze cost sensitivity to noise levels and plot results.
    
    Parameters:
        noise_factors: Array of multipliers for the base noise covariances
        N_runs: Number of simulation runs per noise factor
    """
    # Storage for results
    avg_costs_nom = []
    avg_costs_rob = []
    std_costs_nom = []
    std_costs_rob = []
    
    for factor in noise_factors:
        # Scale noise covariances by current factor
        W_test =  W_hat_list
        V_test = [np.array([[2 + (factor * 10)]])**2]*30 # * 30 for 30 
        #[(np.sqrt(V) + np.array([[noise_factors * 10]]))**2 for V in V_hat_list] # rho = 10.0
        print(f"V_test: {V_test}")
        
        costs_nom = []
        costs_rob = []
        
        for _ in range(N_runs):
            # Sample initial state
            x0 = np.random.multivariate_normal(initial_point.flatten(), X0_hat).reshape(-1, 1)
            
            # Simulate controllers
            _, _, total_cost_nom, _ = simulate_lqg(T, A, B, C, mu, K_list, L_list, x0, W_test, V_test, Q, R)
            _, _, total_cost_rob, _ = simulate_robust_lqg(T, A, B, C, mu, K_list_rob, L_list_rob, x0, W_test, V_test, Q, R)

            
            # Calculate costs
            costs_nom.append(total_cost_nom)
            costs_rob.append(total_cost_rob)
        
        # Store statistics
        avg_costs_nom.append(np.mean(costs_nom))
        avg_costs_rob.append(np.mean(costs_rob))
        std_costs_nom.append(np.std(costs_nom))
        std_costs_rob.append(np.std(costs_rob))
        
        print(f"Noise factor {factor:.2f}: "
              f"Nominal cost = {avg_costs_nom[-1]:.2f} ± {std_costs_nom[-1]:.2f}, "
              f"Robust cost = {avg_costs_rob[-1]:.2f} ± {std_costs_rob[-1]:.2f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot mean costs with error bars
    plt.errorbar(noise_factors, avg_costs_nom, yerr=std_costs_nom, 
                 fmt='o--', color='tab:blue', ecolor='lightblue', elinewidth=3,
                 capsize=5, label='Nominal LQG')
    plt.errorbar(noise_factors, avg_costs_rob, yerr=std_costs_rob,
                 fmt='s-', color='tab:orange', ecolor='moccasin', elinewidth=3,
                 capsize=5, label='Robust LQG')
    
    plt.xlabel('Noise Scaling Factor', fontsize=12)
    plt.ylabel('Total Cost', fontsize=12)
    plt.title('Controller Performance Under Different Noise Levels', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('noise_sensitivity_plot.png', dpi=300)
    plt.show()
    
    return noise_factors, avg_costs_nom, avg_costs_rob


