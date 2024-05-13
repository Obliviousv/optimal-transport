# 验证最优运输问题的解
import numpy as np
from scipy.optimize import linprog

def solve_optimal_transport(cost_matrix, supply, demand):
    # Flatten the cost matrix
    c = cost_matrix.flatten()

    # Define the equality constraints
    A_eq = []
    b_eq = []

    # Add supply constraints
    for i in range(len(supply)):
        constraint = np.zeros_like(c)
        constraint[i * len(demand): (i + 1) * len(demand)] = 1
        A_eq.append(constraint)
        b_eq.append(supply[i])

    # Add demand constraints
    for j in range(len(demand)):
        constraint = np.zeros_like(c)
        constraint[j::len(demand)] = 1
        A_eq.append(constraint)
        b_eq.append(demand[j])

    # Solve the linear programming problem
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, method='simplex')

    # Reshape the solution into a matrix
    solution = np.reshape(res.x, (len(supply), len(demand)))

    return solution

# Example usage
cost_matrix = np.array([[14, 5, 21, 23, 2, 26], 
                        [10, 1, 7, 35, 8, 9], 
                        [16, 13, 27, 6, 11, 12],
                        [15, 20, 18, 36, 19, 17],
                        [22, 28, 3, 24, 25, 4],
                        [29, 30, 31, 32, 33, 34]])
supply = [20, 40, 60, 50, 30, 10]
demand = [22, 34, 41, 27, 17, 69]

solution = solve_optimal_transport(cost_matrix, supply, demand)
print(solution)