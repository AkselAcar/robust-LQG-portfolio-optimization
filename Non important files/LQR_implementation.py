# LQR Trading Strategy Simulation - From the paper by abeille "LQG Porfolio Optimization"

import numpy as np
import matplotlib.pyplot as plt

# Parameters
horizon = 100
lambda_param = 100 #10
sigma = 2 * np.exp(1) - 2
eta = 0.5
gamma_i = 5 #0.06
omega_p = 0.1
omega_i = 0.2
beta_i = -0.0048

# Define system matrices
A = np.array([
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, eta, eta - 1],
    [0, 0, 0, 1 - omega_p, 0],
    [0, 0, 0, 0, 1 - omega_i]
])

B = np.array([[1], [gamma_i], [gamma_i * (eta - 1)], [0], [omega_i * beta_i]])

Q = np.array([
    [lambda_param * sigma**2, 0, 0, -1/2, -1/2],
    [0, 0, 0, -1/2, -1/2],
    [0, 0, 0, 0, 0],
    [-1/2,0, 0, 0, 0],
    [-1/2, 0, 0, 0, 0]
])

R = np.array([[lambda_param * eta**2 * sigma**2 - gamma_i * (eta - 1)]]) 

N = np.array([[lambda_param * eta * sigma**2 - (1/2) * gamma_i], [0], [0], [0], [0]])


def solve_finite_horizon_riccati(horizon):
    """ Solves the finite-horizon Riccati equation and returns feedback gains K. """

    # Check invertibility
    if np.linalg.det(R) == 0:
        raise ValueError("Matrix R is singular and cannot be inverted.")

    # Riccati recursion
    P = np.zeros((horizon + 1, *Q.shape))
    K = np.zeros((horizon, B.shape[1], A.shape[0]))  # Feedback gain over time

    P[horizon] = Q  # Terminal condition
    
    for t in range(horizon - 1, -1, -1):
        try:
            S = R + B.T @ P[t + 1] @ B
            if S[0,0] <= 0:
                raise ValueError(f"Singular matrix encountered at time step {t}")
            
            K[t] = np.linalg.inv(S) @ (B.T @ P[t + 1] @ A + N.T)  # be careful with the transpose
            P[t] = Q + A.T @ P[t + 1] @ A - K[t].T @ S @ K[t]

        except np.linalg.LinAlgError:
            raise ValueError(f"Failed to compute inverse at time step {t}")

    return K  # Return computed feedback gains


# ============================
# Simulating the LQR Trading Strategy
# ============================
def simulate_trading(K, horizon, initial_state):
    """ Simulates the LQR trading strategy over the given time horizon. """

    # Initialize state and control history
    x = np.zeros((horizon, A.shape[0], 1))  # State vector over time
    q = np.zeros((horizon, 1, 1))  # Trade sizes (corrected to (1,1) shape) # basically our input
    pnl = np.zeros(horizon)  # Profit and Loss over time

    x[0] = initial_state  # Set initial state

    for t in range(horizon - 1):
        q[t] = -K[t] @ x[t]  # Compute optimal trade (ensure q has correct shape)
        x[t + 1] = A @ x[t] + B @ q[t]  # Update state

        # Compute PnL as execution return times trade quantity
        #pnl[t] = x[t][0]*x[t][1] + q[t] * x[t][2]  # Ensuring scalar multiplication
        
        pnl[t] = (x[t][0] + q[t]) * x[t][2]
    
    return x, q, pnl


# ============================
# Running the Simulation
# ============================

# Compute LQR gains
K = solve_finite_horizon_riccati(horizon)

# Initial state (e.g., starting inventory, initial price, etc.)
initial_state = np.array([[10],  # Initial inventory (Q)
                          [0],   # Initial predictable return (x^p)
                          [0],   # Initial execution return (x^exe)
                          [1],   # Initial decision return (x^dec)
                          [0]])  # Initial impact memory (x^i)

# Simulate the trading strategy
x, q, pnl = simulate_trading(K, horizon, initial_state)

# ============================
# Plot Results
# ============================
plt.figure(figsize=(12, 6))

# Plot trading decisions
plt.subplot(2, 1, 1)
plt.plot(x[:, 0, 0], label=r"Inventory \( Q_t \)", color='blue')
plt.xlabel("Time")
plt.ylabel("Q_t")
plt.title("Optimal Trading Strategy (LQR Control)")
plt.legend()

# Plot cumulative PnL
plt.subplot(2, 1, 2)
plt.plot(np.cumsum(pnl), label="Cumulative PnL", color='red')
plt.xlabel("Time")
plt.ylabel("Profit and Loss")
plt.title("Cumulative Trading PnL")
plt.legend()

plt.tight_layout()
plt.show()
