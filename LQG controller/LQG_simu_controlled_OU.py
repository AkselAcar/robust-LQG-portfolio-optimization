# Code that computes the LQG controller

import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
T = 20.0              # total time
dt = 0.01             # time step
N = int(T / dt)       # number of steps
time = np.linspace(0, T, N)

# --- Model Parameters ---
theta = 0.15           # mean-reversion rate
mu = 130.0            # target portfolio value
sigma = 2.0           # process noise strength
measurement_noise_std = 1.0  # output (observation) noise
R = 15                # control cost
Q = 10                # state deviation cost

# --- Transformed equation coefficients ---
a = 1 - theta * dt    # Euler discretization
b = -dt
c = 1.0               # Observation model (z directly observed with noise)

# --- Solve Discrete-Time Riccati Equation for scalar system (Control - LQR) ---
P = Q
for _ in range(1000):
    P = Q + a**2 * P - (a * b * P)**2 / (R + b**2 * P)
K = (a * b * P) / (R + b**2 * P)

# --- Solve Discrete-Time Riccati Equation for Kalman Filter (Estimation) ---
Qw = sigma**2 * dt               # process noise covariance
Rv = measurement_noise_std**2    # measurement noise covariance
Sigma = 1.0                      # Initial error covariance
for _ in range(1000):
    Sigma = a**2 * Sigma + Qw - (a * Sigma * c)**2 / (c**2 * Sigma + Rv)
L = a * Sigma * c / (c**2 * Sigma + Rv)  # Kalman gain

# --- Initialization ---
x = np.zeros(N)         # True state (portfolio value)
z = np.zeros(N)         # True transformed state: z_k = mu - x_k
u = np.zeros(N)         # Control input
cost = np.zeros(N)      # Cost
y = np.zeros(N)         # Noisy measurement of z
z_hat = np.zeros(N)     # Estimated z from Kalman filter

x[0] = 120.0
z[0] = mu - x[0]
z_hat[0] = z[0] + np.random.normal(0, measurement_noise_std)

# --- Simulation Loop ---
for k in range(1, N):
    # Control based on estimated state
    u[k] = -K * z_hat[k-1]

    # Process noise
    w_k = np.random.normal(0, sigma * np.sqrt(dt))

    # True system evolution
    z[k] = a * z[k-1] + b * u[k] + w_k
    x[k] = mu - z[k]

    # Noisy measurement
    v_k = np.random.normal(0, measurement_noise_std)
    y[k] = c * z[k] + v_k

    # Kalman filter update
    z_hat[k] = a * z_hat[k-1] + b * u[k] + L * (y[k] - c * (a * z_hat[k-1] + b * u[k]))

    # Instantaneous cost
    cost[k] = Q * z[k-1]**2 + R * u[k]**2

# --- Plot Portfolio Value ---
plt.figure(figsize=(12, 6))
plt.plot(time, x, label='True Portfolio Value $x_k$', linewidth=2)
plt.plot(time, mu - z_hat, label='Estimated Portfolio Value $\hat{x}_k$', linestyle='--')
plt.axhline(mu, color='gray', linestyle='--', label='Target $\\mu$')
plt.title("LQG-Controlled Market evolution with Measurement Noise")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Control Input ---
plt.figure(figsize=(12, 4))
plt.plot(time, u, label='Control Input $u_k$', color='orange')
plt.title("Control Input Over Time (LQG)")
plt.xlabel("Time")
plt.ylabel("Control $u_k$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Measurement vs True State ---
plt.figure(figsize=(12, 4))
plt.plot(time, z, label='True Deviation $z_k$')
plt.plot(time, y, label='Noisy Measurement $y_k$', alpha=0.6)
plt.plot(time, z_hat, label='Estimated $\\hat{z}_k$', linestyle='--')
plt.title("True vs Measured vs Estimated State")
plt.xlabel("Time")
plt.ylabel("Deviation from Target")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
