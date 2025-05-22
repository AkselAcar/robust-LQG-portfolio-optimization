import numpy as np

def check_star_condition(horizon, lambda_param, sigma, eta, gamma_i, omega_p, omega_i, beta_i):
    """
    Checks whether the * condition holds using the given formulation:
    [H, I]^T * [(Q, N); (N, R)] * [H, I] > 0.
    
    Args:
        horizon (int): Horizon length.
        lambda_param (float): Risk tuning parameter lambda.
        sigma (float): Variance term.
        eta (float): Model parameter.
        gamma_i (float): Model parameter.
        omega_p (float): Model parameter.
        omega_i (float): Model parameter.
        beta_i (float): Model parameter.

    Returns:
        bool: True if the condition is satisfied, False otherwise.
    """
    # Define matrices from the document
    A = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, eta, eta - 1],
        [0, 0, 0, 1 - omega_p, 0],
        [0, 0, 0, 0, 1 - omega_i]
    ])

    B = np.array([[1], [gamma_i], [gamma_i * (eta - 1)], [0], [omega_i * beta_i]])
    
    Q = np.array([
        [lambda_param * sigma, 0, 0, -1/2, -1/2],
        [0, 0, 0, -1/2, -1/2],
        [0, 0, 0, 0, 0],
        [-1/2, -1/2, 0, 0, 0],
        [-1/2, -1/2, 0, 0, 0]
    ])

    R = np.array([[lambda_param * eta**2 * sigma**2 - gamma_i * (eta - 1)]])
    
    N = np.array([[lambda_param*eta * sigma**2 - (1/2)*gamma_i], [0], [0], [0], [0]]).T  
    
    Q_blk_diag = np.kron(np.eye(horizon), Q)
    R_blk_diag = np.kron(np.eye(horizon), R)
    N_blk_diag = np.kron(np.eye(horizon), N)    

    print("shape of Q:", Q.shape)
    print("shape of R:", R.shape)
    print("shape of N:", N.shape)
    print("shape of Q_blk_diag:", Q_blk_diag.shape)
    print("shape of R_blk_diag:", R_blk_diag.shape)
    print("shape of N_blk_diag:", N_blk_diag.shape)

    # Construct the block matrix M = [(Q, N); (N, R)]
    M = np.block([
        [Q_blk_diag, N_blk_diag.T],
        [N_blk_diag, R_blk_diag]
    ])
    
    # Construct H and I matrices !!! Problematic matrix !!! dimensions do not match 
    H = np.array([[0 ,0, 0], 
                  [B, 0, 0], 
                  [A@B, B, 0]])
    
    I = np.eye(R.shape[0]*horizon)  # Identity matrix for I
    
    HI = np.vstack([H, I])  # [H, I]
    
    # Compute the star condition matrix
    star_matrix = HI.T @ M @ HI
    
    # Check if the matrix is positive definite
    eigenvalues = np.linalg.eigvals(star_matrix)
    
    if np.all(eigenvalues > 0):
        print("* Condition is satisfied.")
        return True
    else:
        print("* Condition is NOT satisfied.")
        return False

# Example usage
check_star_condition(horizon=3,lambda_param=0.5,sigma=2*np.exp(1)-2,eta=0.5,gamma_i=0.06,omega_p=0.1,omega_i=0.2,beta_i=-0.0048)
