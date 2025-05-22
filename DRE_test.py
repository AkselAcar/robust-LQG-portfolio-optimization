import numpy as np

def solve_finite_horizon_riccati(horizon, lambda_param, sigma, eta, gamma_i, omega_p, omega_i, beta_i):
    """
    Solves the finite-horizon Riccati difference equation for given parameters.
    
    Args:
        horizon (int): Horizon length.
        lambda_param (float): Risk tuning parameter lambda.
        sigma (float): Variance term.
        eta (float): Model parameter.
        gamma_i (float): Model parameter.
        omega_p (float): Model parameter.
        omega_i (float): Model parameter.
        beta_i (float): Model parameter.
    
    Raises:
        ValueError: If the computation fails due to matrix inversion issues.
    """ 

    # Define matrices from page 12 of the document
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
        [-1/2, 0, 0, 0, 0],
        [-1/2, 0, 0, 0, 0]
    ])

    R = np.array([[lambda_param * eta**2 * sigma**2 - gamma_i * (eta - 1)]]) 

    N = np.array([[lambda_param * eta * sigma**2 - (1/2) * gamma_i], [0], [0], [0], [0]]).T

    # Check invertibility
    if np.linalg.det(R) == 0:
        raise ValueError("Matrix R is singular and cannot be inverted.")

    # Riccati recursion
    P = np.zeros((horizon + 1, *Q.shape))
    K = np.zeros((horizon, B.shape[1], A.shape[0]))  # Stores feedback gains over time

    P[horizon] = Q  # Terminal condition
    
    for t in range(horizon - 1, -1, -1):
        try:
            S = R + B.T @ P[t + 1] @ B
            #print(f"S[{t}] =\n{S}")
            if S[0,0] <= 0:
                raise ValueError(f"Singular matrix encountered at time step {t}")
            
            K[t] = np.linalg.inv(S) @ (B.T @ P[t + 1] @ A + N)
            #print(f"K[{t}] =\n{K[t]}")
            P[t] = Q + A.T @ P[t + 1] @ A - K[t].T @ S @ K[t]
            #print(f"P[{t}] =\n{P[t]}")


        
        except np.linalg.LinAlgError:
            raise ValueError(f"Failed to compute inverse at time step {t}")

    print("Computation successful.")


for lambda_param in np.linspace(0.00064, 1000, 100):
    print(f"lambda_param = {lambda_param}")
    solve_finite_horizon_riccati(
        horizon=1000,
        lambda_param=lambda_param,
        sigma=2*np.exp(1)-2,
        eta=0.5,
        gamma_i=0.06,
        omega_p=0.1,
        omega_i=0.2,
        beta_i=-0.0048
    )

        
 