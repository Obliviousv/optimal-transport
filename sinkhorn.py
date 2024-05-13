# sinkhorn法求解最优传输问题
import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=np.inf)
def sinkhorn(a, b, M, reg, num_iters):
    Err_p = np.zeros(num_iters)
    Err_q = np.zeros(num_iters)
    scale = np.sum(a)
    a = a / scale
    b = b / scale
    K = np.exp(-M / reg)
    # Initialize the scaling factors
    u = np.ones_like(a)
    v = np.ones_like(b)

    # Iterate Sinkhorn's algorithm
    for i in range(num_iters):
        u = np.divide(a, K@v)
        v = np.divide(b, K.T@u)
        # Compute the optimal transport plan
        P = np.diag(u) @ K @ np.diag(v)*scale
        Err_p[i] = np.linalg.norm(P @ np.ones_like(a) - a*scale, ord=1)
        Err_q[i] = np.linalg.norm(P.T @ np.ones_like(b) - b*scale, ord=1)

    return P, Err_p, Err_q


cost_matrix = np.array([[14, 5, 21, 23, 2, 26], 
                        [10, 1, 7, 35, 8, 9], 
                        [16, 13, 27, 6, 11, 12],
                        [15, 20, 18, 36, 19, 17],
                        [22, 28, 3, 24, 25, 4],
                        [29, 30, 31, 32, 33, 34]])
supply = [20, 40, 60, 50, 30, 10]
demand = [22, 34, 41, 27, 17, 69]
eps = 0.1
num_iters = 2000  # Number of iterations

P,Err_p,Err_q = sinkhorn(supply, demand, cost_matrix, eps, num_iters)
print(f"transport plan: {P}")
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.title("$||P1 -a||_1$")
plt.plot(np.log(Err_p), linewidth=1)
plt.subplot(2, 1, 2)
plt.title("$||P^T 1 -b||_1$")
plt.plot(np.log(Err_q), linewidth=1)
plt.show()