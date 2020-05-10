import matplotlib.pyplot as plt
import numpy as np
import math
import time

N = 10
U = 230.0
#c = np.array([1.0 for _ in range(N)]) # coût constant
#c = np.array([2.0*(math.cos(i) + 1.1) for i in range(N)]) # coût oscillant
c = np.array([(2.0*(i%3) + 1.0) for i in range(N)]) # coût "triangulaire"

Qf1 = 15.0
Qi1 = 2.0
Qf2 = 13.0
Qi2 = 7.0
tf1 = 24.0
ti1 = 7.0
tf2 = 17.0
ti2 = 9.0
P_max = max(1.5*(Qf1-Qi1)/(tf1 - ti1) * U, 1.5*(Qf2-Qi2)/(tf2 - ti2) * U)

In_plus = [[0 if x!=k else 1 for x in range(N)] for k in range(N)]
In_moins = [[0 if x!=k else -1 for x in range(N)] for k in range(N)]
A1 = np.array([[-(tf1 - ti1)/(N*U)]*N] + In_moins + In_plus)
A2 = np.array([[-(tf2 - ti2)/(N*U)]*N] + In_moins + In_plus)

def funi(x):
    return np.dot(c, x)

def grad_funi(x):
    return c

def grad_ci(x):
    return A.T

def cont1(x):
    c1 = [(Qf1 - Qi1) - (tf1 - ti1)*sum(y for y in x)/(N * U)]
    c2 = [-y for y in x]
    c3 = [y - P_max for y in x]
    return np.array(c1 + c2 + c3)

def cont2(x):
    c1 = [(Qf2 - Qi2) - (tf2 - ti2)*sum(y for y in x)/(N * U)]
    c2 = [-y for y in x]
    c3 = [y - P_max for y in x]
    return np.array(c1 + c2 + c3)

def uzawa(xk1, xk2, grad_f1, grad_f2, grad_l1, grad_l2, c1, c2):
    # 1 Initialisation
    lambdak = [0]*(4*N + 2)
    rho = 0.5
    # 2 Décomposition
    def decomp(xk, lk, grad_f, grad_c, alpha=1e-2, maxit=1e3, eps=1e-8):
        i = 0
        grad_l_xk = grad_l(xk, lk)
        while (i < maxit) and (np.linalg.norm(grad_l_xk > eps)):
            i += 1
            grad_l_xk = grad_f(xk, lk) + np.dot(grad_c(xk, lk), lk)
            xk = xk - alpha*grad_l_xk
        return xk
    x1 = decomp(x1, lambdak[0:2*N+1], )