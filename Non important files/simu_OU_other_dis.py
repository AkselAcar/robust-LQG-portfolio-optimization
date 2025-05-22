# Old version of the code for simulating a portfolio with LQR control

import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
T = 100.0             # total time
dt = 0.1             # time step
N = int(T / dt)       # number of steps
time = np.linspace(0, T, N)

# --- Model Parameters ---
theta = 0.1    #0.5   # smaller mean-reversion rate for natural stability
mu = 150.0            # target portfolio value
sigma = 2.0           # noise strength
R = 20         #0.1   # control cost
Q = 10         #1     # state deviation cost

# --- Adjusted Discrete-time Dynamics ---
# Set a to be closer to 1 for more stability (less overshoot)
a = 1 - theta * dt    # equivalent to Euler discretization of -theta*(mu - x)
b = -dt               # control input scaling (still applied to z_k)
# In this form: z_{k+1} = a * z_k + b * u_k + w_k

# --- Solve Discrete-Time Riccati Equation for scalar system ---
P = Q
for _ in range(1000):  # Iterate until convergence
    P = Q + a**2 * P - (a * b * P)**2 / (R + b**2 * P)

# --- Compute LQR Gain ---
K = (a * b * P) / (R + b**2 * P)

# --- Initialization ---
x = np.zeros(N)
z = np.zeros(N)           # z_k = mu - x_k
u = np.zeros(N)
cost = np.zeros(N)

x[0] = 120.0              # Initial portfolio value
z[0] = mu - x[0]          # Initial deviation from target

# --- Simulation Loop ---
for k in range(1, N):
    # Optimal control
    u[k] = -K * z[k-1]

    # Process noise
    w_k = np.random.normal(0, sigma * np.sqrt(dt))

    # Update z_k using stable dynamics
    z[k] = a * z[k-1] + b * u[k] + w_k

    # Recover portfolio value
    x[k] = mu - z[k]

    # Cost at time step
    cost[k] = Q * z[k-1]**2 + R * u[k]**2

# --- Plot Portfolio Value ---
plt.figure(figsize=(12, 6))
plt.plot(time, x, label='Portfolio Value $x_k$', linewidth=2)
plt.axhline(mu, color='gray', linestyle='--', label='Target $\\mu$')
plt.title("Smoothed LQR-Controlled Portfolio (Stable Natural Drift)")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Cost Over Time ---
plt.figure(figsize=(12, 4))
plt.plot(time, cost, label='Instantaneous Cost')
plt.title("LQR Cost Over Time")
plt.xlabel("Time")
plt.ylabel("Cost $Q(\mu - x_k)^2 + R u_k^2$")
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()
plt.show()
