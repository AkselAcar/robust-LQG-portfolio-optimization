import numpy as np
from LQG_controller import compute_finite_horizon_lqr_gains, finite_horizon_kalman_gains
from simulate_lqg import simulate_lqg, simulate_robust_lqg
from simulation_plots import simulate_and_plot_average_trajectories
from noise_cost_sensitivity import analyze_noise_sensitivity
from save_sdp_results import compute_and_save_sdp_results, load_sdp_results
from save_sdp_results import compute_and_save_sdp_results, load_sdp_results, should_recompute_sdp

import os

# --- Parameters ---
time_horizon = 30.0  # Total time 
dt = 1.0   # Time step
T = int(time_horizon / dt)  # Number of steps 

# System dimensions
n = 1  # Number of states, to chnage if needed !!
m = 1  # Number of control inputs

# --- System Matrices ---
theta = np.diag([0.1])        # mean reversion coefficient for each state, to adapt if needed !!
A = np.eye(n) - theta * dt     # Dynamics matrix
B = -dt * np.ones((n, m))      # Control input matrix
C = np.eye(n)                  # Measurement matrix

# --- Initial State and Noise Covariances ---
mu = np.array([[130.0]])             # To adapt if needed !! Multiple dimensions possible !
sigma_x0 = 2.0                       # Initial state uncertainty
X0_hat = sigma_x0**2 * np.eye(n)     # Initial state covariance (n x n)
sigma_w = 2.0                        # Process noise intensity
W_hat = sigma_w**2 * dt * np.eye(n)  # Process noise covariance (n x n)
sigma_v = 2.0                        # Measurement noise intensity
V_hat = sigma_v**2 * np.eye(n)       # Measurement noise covariance (p x p)

# --- Cost Matrices ---
Q = 4 * np.eye(n)  # State cost
R = 2 * np.eye(m)  # Control cost


# --- SDP Parameters ---
sdp_params = {
    "A": A,
    "B": B,
    "C": C,
    "X0_hat": X0_hat,
    "W_hat": W_hat,
    "V_hat": V_hat,
    "dt": dt,
    "T": T,
    "rho_x0": 0.0,
    "rho_w": 0.0,
    "rho_v": 10.0
}

# --- Compute or load SDP results ---
SDP_FILENAME = "sdp_results.npz"

# Check if we need to recompute SDP
if should_recompute_sdp(sdp_params, SDP_FILENAME):
    print("Parameters changed or file missing. Computing new SDP solution...")
    sdp_result = compute_and_save_sdp_results(sdp_params, SDP_FILENAME)
    X0_robust = sdp_result["X0"]
    W0_robust = sdp_result["W0"]
    V0_robust = sdp_result["V0"]
else:
    print("Loading existing SDP solution...")
    saved_data = load_sdp_results(SDP_FILENAME)
    X0_robust = saved_data["X0_robust"]
    W0_robust = saved_data["W0_robust"]
    V0_robust = saved_data["V0_robust"]
    print(f"Loaded SDP solution computed with parameters: {saved_data['params']}")

# print("X0_hat eigenvalues:", np.linalg.eigvals(X0_hat))
# print("W_hat eigenvalues:", np.linalg.eigvals(W_hat))
# print("V_hat eigenvalues:", np.linalg.eigvals(V_hat))

# --- Compute Nominal and Robust Gains ---
K_list = compute_finite_horizon_lqr_gains(A, B, Q, R, T=T)
L_list = finite_horizon_kalman_gains(A, C, W_hat, V_hat, X0_hat, T)
L_rob = finite_horizon_kalman_gains(A, C, W0_robust, V0_robust, X0_robust, T)

# --- Simulation Parameters ---
noise_factor = 0.95
W_real =  W0_robust
V_real = [noise_factor * V for V in V0_robust]
initial_point = 0.80 * mu
N_runs = 100

# --- Run Simulations and Plot Results ---
x_nom_avg, x_rob_avg, avg_cost_nom, avg_cost_rob = simulate_and_plot_average_trajectories(
    T, A, B, C, mu,
    K_list, L_list,
    K_list_rob=K_list,     
    L_list_rob=L_rob,
    X0_robust=X0_robust,
    W_test=W_real,
    V_test=V_real,
    initial_point=initial_point,
    N_runs=N_runs,
    Q=Q,
    R=R
)

# Print final cost comparison
print(f"\n=== Final Results ===")
print(f"Nominal Controller Average Cost: {avg_cost_nom}")
print(f"Robust Controller Average Cost: {avg_cost_rob}")
print(f"Robustness Benefit: {avg_cost_nom - avg_cost_rob}")

# --- Noise Sensitivity Analysis ---
W_hat_list = [W_hat for _ in range(T)]
V_hat_list = [V_hat for _ in range(T)]

noise_factors, avg_nom, avg_rob = analyze_noise_sensitivity(
    T=T,
    A=A,
    B=B,
    C=C,
    mu=mu,
    K_list=K_list,
    L_list=L_list,
    K_list_rob=K_list,
    L_list_rob=L_rob,
    X0_hat=X0_hat,
    W_hat_list=W_hat_list,
    V_hat_list=V_hat_list,
    initial_point=initial_point,
    Q=Q,
    R=R,
    noise_factors=np.linspace(0, 1.25, 11),
    N_runs=25,
    rho_v=sdp_params["rho_v"]
)





