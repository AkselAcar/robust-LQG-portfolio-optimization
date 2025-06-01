# New version of the LQG simulation script

import numpy as np
from src.LQG_controller import compute_lqr_gain, finite_horizon_kalman_gains
from src.simulate_lqg import simulate_lqg, simulate_robust_lqg
from src.sdp_solver import solve_sdp

# --- Parameters ---
time_horizon = 30.0  # Total time 
dt = 1.0   # Time step
T = int(time_horizon / dt)  # Number of steps 

# System dimensions
n = 2  # Number of states
m = 1  # Number of control inputs

# --- System Matrices ---
theta = np.diag([0.15,0.1])  #	mean reversion coefficient for each state, add target portfolio value for each state !!!!
A = np.eye(n) - theta * dt     # Dynamics matrix (n x n)
B = -dt * np.ones((n, m))      # Control input matrix (n x m)
C = np.eye(n)               # Measurement matrix (p x n) (measures first state)

# --- Initial State and Noise Covariances ---
mu = np.array([[130.0], [130.0]])  # add target portfolio value for each state !!!!
sigma_x0 = 2.0                     # Initial state uncertainty
X0_hat = sigma_x0**2 * np.eye(n)   # Initial state covariance (n x n)
sigma_w = 2.0                      # Process noise intensity
W_hat = sigma_w**2 * dt * np.eye(n) 	# Process noise covariance (n x n)
sigma_v = 2.0                       # Measurement noise intensity
V_hat = sigma_v**2 * np.eye(n)      	# Measurement noise covariance (p x p) 

# --- Cost Matrices ---
Q = 10 * np.eye(n)  # State cost (penalize state deviations)
R = 15 * np.eye(m)  # Control cost (penalize large inputs)

# --- Compute Nominal LQR and Kalman Gains ---
K, _ = compute_lqr_gain(A, B, Q, R)          # LQR gain (m x n)
L_list, _ = finite_horizon_kalman_gains(A, C, W_hat, V_hat, X0_hat, T)

# --- Solve SDP for Robust Covariances ---
sdp_result = solve_sdp(A, B, C, X0_hat, W_hat, V_hat, dt, T)
X0_robust = sdp_result["X0"]  # Robust initial covariance (n x n)
W0_robust = sdp_result["W0"]  # Robust process noise covariance (n x n)
V0_robust = sdp_result["V0"]  # Robust measurement noise covariance (p x p)

# --- Compute Robust Kalman Gain ---
L_rob, _ = finite_horizon_kalman_gains(A, C, W0_robust, V0_robust, X0_robust, T) 

