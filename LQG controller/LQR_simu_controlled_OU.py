# LQR Trading Strategy Simulation
# This code simulates a trading strategy using Linear Quadratic Regulator (LQR) control. 
# Also compares the market evolution with the mean reversion and with the controlled mean reversion.
""" import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
T = 10.0              # total time
dt = 0.01             # time step
N = int(T / dt)       # number of steps
time = np.linspace(0, T, N)

# --- Model Parameters ---
theta = 0.1           # mean-reversion rate
mu = 100.0            # target portfolio value
sigma = 2.0           # noise strength
R = 100               # control cost
Q = 10              # state deviation cost

# --- Transformed equation coefficients ---
a = 1 - theta * dt    # equivalent to Euler discretization of -theta*(mu - x)
b = -dt

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

    # Update z_k using transformed dynamics
    z[k] = a * z[k-1] + b * u[k] + w_k

    # Recover portfolio value
    x[k] = mu - z[k]

    # Cost at time step
    cost[k] = Q * z[k-1]**2 + R * u[k]**2

# --- Plot Portfolio Value ---
plt.figure(figsize=(12, 6))
plt.plot(time, x, label='Portfolio Value $x_k$', linewidth=2)
plt.axhline(mu, color='gray', linestyle='--', label='Target $\\mu$')
plt.title("LQR-Controlled Portfolio with Transformed OU Dynamics")
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
plt.grid(True)
plt.tight_layout()
plt.show()
 """


# Re-import necessary libraries after environment reset
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
sigma = 2.0           # noise strength
R = 15                # control cost
Q = 10                # state deviation cost

# --- Transformed equation coefficients ---
a = 1 - theta * dt    # Euler discretization
b = -dt

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
z_no_control = np.zeros(N)
x_no_control = np.zeros(N)

x[0] = 120.0              # Initial portfolio value
z[0] = mu - x[0]          # Initial deviation from target
z_no_control[0] = z[0]  # Initial value for no control case
x_no_control[0] = x[0]  # Initial portfolio value for no control case

# --- Simulation Loop ---
for k in range(1, N):
    # Optimal control
    u[k] = -K * z[k-1]

    # Process noise
    w_k = np.random.normal(0, sigma * np.sqrt(dt))

    # Update z_k using transformed dynamics
    z[k] = a * z[k-1] + b * u[k] + w_k  # Adjusted for previous control input

    z_no_control[k] = a * z_no_control[k-1] + w_k  # No control case

    # Recover portfolio value
    x[k] = mu - z[k]

    x_no_control[k] = mu - z_no_control[k]  # No control case

    # Cost at time step
    cost[k] = Q * z[k-1]**2 + R * u[k]**2

# --- Plot Portfolio Value ---
plt.figure(figsize=(12, 6))
plt.plot(time, x, label='Market Value $x_k$', linewidth=2)
plt.axhline(mu, color='gray', linestyle='--', label='Target $\\mu$')
plt.title("LQR-Controlled Market evolution with OU Dynamics")
plt.xlabel("Time")
plt.ylabel("Market Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Create subplots
fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Plot 1: Portfolio Value ---
ax[0].plot(time, x, label='Market Value $x_k$', linewidth=2)
ax[0].axhline(mu, color='gray', linestyle='--', label='Target $\\mu$')
ax[0].set_title("LQR-Controlled Market evolution with OU Dynamics")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Market Value")
ax[0].legend()
ax[0].grid(True)

# --- Plot 2: (Example) Control Input ---
ax[1].plot(time, x_no_control, label='Market value $x_k$', color='orange', linewidth=2)
ax[1].axhline(mu, color='gray', linestyle='--', label='Target $\\mu$')
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Market Value")
ax[1].set_title("Market Value without Control")
ax[1].legend()
ax[1].grid(True)



# --- Plot Cost Over Time ---
plt.figure(figsize=(12, 4))
plt.plot(time, cost, label='Instantaneous Cost')
plt.title("LQR Cost Over Time")
plt.xlabel("Time")
plt.ylabel("Cost $Q(\mu - x_k)^2 + R u_k^2$")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Control Input Over Time ---
plt.figure(figsize=(12, 4))
plt.plot(time, u, label='Control Input $u_k$', color='orange')
plt.title("Control Input Over Time")
plt.xlabel("Time")
plt.ylabel("Control $u_k$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
