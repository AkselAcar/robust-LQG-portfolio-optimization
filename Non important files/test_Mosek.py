# To learn more about MOSEK and how to write Python code for it. 


# import cvxpy as cp
# import numpy as np

# # Define variables
# x = cp.Variable(2)

# # Quadratic objective: (1/2) * x^T * P * x + q^T * x
# P = np.array([[4, 1],
#               [1, 2]])
# q = np.array([1, 1])

# # Constraints: Ax <= b
# A = np.array([[1, 1],
#               [1, -1]])
# b = np.array([1, 0.2])

# # Define the problem
# objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
# constraints = [A @ x <= b]
# problem = cp.Problem(objective, constraints)

# # Solve using MOSEK
# problem.solve(solver=cp.MOSEK)

# # Results
# print("Status:", problem.status)
# print("Optimal value:", problem.value)
# print("Optimal x:", x.value)


import cvxpy as cp
import numpy as np

# Define symmetric matrix variable (2x2)
X = cp.Variable((2, 2), symmetric=True)

# Define data
C = np.array([[1, 0],
              [0, 0]])
A1 = np.array([[1, 1],
               [1, 0]])
A2 = np.array([[0, 1],
               [1, 1]])
b1 = 1
b2 = 1

# Objective: minimize Tr(CX)
objective = cp.Minimize(cp.trace(C @ X))

# Constraints
constraints = [
    cp.trace(A1 @ X) == b1,
    cp.trace(A2 @ X) == b2,
    X >> 0  # X is positive semidefinite
]

# Define problem
problem = cp.Problem(objective, constraints)

# Solve
problem.solve(solver=cp.MOSEK)

# Output results
print("Status:", problem.status)
print("Optimal value:", problem.value)
print("Optimal X:")
print(X.value)
