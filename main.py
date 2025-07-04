import numpy as np
from LQG_controller import compute_finite_horizon_lqr_gains, finite_horizon_kalman_gains
from simulation_plots import simulate_and_plot_average_trajectories
from noise_cost_sensitivity import analyze_noise_sensitivity
from save_sdp_results import compute_and_save_sdp_results, load_sdp_results
from save_sdp_results import compute_and_save_sdp_results, load_sdp_results, should_recompute_sdp

# --- Parameters ---
time_horizon = 30.0  # Total time 
dt = 1.0   # Time step
T = int(time_horizon / dt)  # Number of steps 

# System dimensions
n = 2 # Number of states, to change for multiple dimensions !
m = 1  # Number of control inputs

# --- System Matrices ---
theta = np.diag([0.1, 0.05])   # Mean reversion coefficient for each state, to change for multiple dimensions !
A = np.eye(n) - theta * dt     # Dynamics matrix
B = -dt * np.ones((n, m))      # Control input matrix
C = np.eye(n)                  # Measurement matrix

# --- Initial State and Noise Covariances ---
mu = np.array([[130.0], [150.0]])    # To change for multiple dimensions !
sigma_x0 = 2.0                       # Initial state uncertainty
X0_hat = sigma_x0**2 * np.eye(n)     # Initial state covariance (n x n)
sigma_w = 2.0                        # Process noise intensity
W_hat = sigma_w**2 * dt * np.eye(n)  # Process noise covariance (n x n)
sigma_v = 2.0                        # Measurement noise intensity
V_hat = sigma_v**2 * np.eye(n)       # Measurement noise covariance (p x p)

# --- Cost Matrices ---
Q = 2 * np.eye(n)  # State cost
R = 1 * np.eye(m)  # Control cost

# --- Convert scalar covariances to time-varying lists ---
W_hat_list = [W_hat for _ in range(T)]  # Create list of identical W_hat matrices
V_hat_list = [V_hat for _ in range(T)]  # Create list of identical V_hat matrices

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
    "rho_x0": 20.0,
    "rho_w": 0.0,  # Process noise robustness parameter, if too big blows up the variance in simulations-> to investigate
    "rho_v": 20.0
}

# --- Compute or load SDP results ---
SDP_FILENAME = "sdp_results.npz"

# Check if we need to recompute SDP
if should_recompute_sdp(sdp_params, SDP_FILENAME):
    print("Parameters changed or file missing. Computing new SDP solution...")
    sdp_result = compute_and_save_sdp_results(sdp_params, SDP_FILENAME)
    X0_robust = sdp_result["X0"]
    W_list_robust = sdp_result["W_list"]  # Make sure this matches your SDP return keys
    V_list_robust = sdp_result["V_list"]  # Make sure this matches your SDP return keys
else:
    print("Loading existing SDP solution...")
    saved_data = load_sdp_results(SDP_FILENAME)
    X0_robust = saved_data["X0_robust"]
    W_list_robust = saved_data["W_list_robust"]  
    V_list_robust = saved_data["V_list_robust"]  
    print(f"Loaded SDP solution computed with parameters: {saved_data['params']}")

print("W_list_robust: ", W_list_robust)
print("V_list_robust: ", V_list_robust) 

# --- Compute Nominal and Robust Gains ---
K_list = compute_finite_horizon_lqr_gains(A, B, Q, R, T=T)
L_list = finite_horizon_kalman_gains(A, C, W_hat_list, V_hat_list, X0_hat, T)  
L_rob = finite_horizon_kalman_gains(A, C, W_list_robust, V_list_robust, X0_robust, T)


# --------------------------------------------- Simulation Setup ---------------------------------------------


# --- Simulation Parameters ---
noise_factor = 0.95 # Factor to scale process noise for robustness, to adapt as wanted
W_real = [noise_factor * W for W in W_list_robust]         
V_real = [noise_factor * V for V in V_list_robust]
#initial point with noise
initial_point = np.random.multivariate_normal(mu.flatten(), X0_hat).reshape(-1, 1) 
N_runs = 100 # Number of simulation runs

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

# --- Noise Sensitivity Analysis ---
# These are already created above, but keeping for consistency
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
    W_list_robust=W_list_robust,
    V_list_robust=V_list_robust,
    initial_point=initial_point,
    Q=Q,
    R=R,
    noise_factors=np.linspace(0, 1.2, 11),
    N_runs=25,
)