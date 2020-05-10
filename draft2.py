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
Pmax = max(1.5*(Qf1-Qi1)/(tf1 - ti1) * U, 1.5*(Qf2-Qi2)/(tf2 - ti2) * U)

In_plus = [[0 if x!=k else 1 for x in range(N)] for k in range(N)]
In_moins = [[0 if x!=k else -1 for x in range(N)] for k in range(N)]
A1 = np.array([[-(tf1 - ti1)/(N*U)]*N] + In_moins + In_plus)
A2 = np.array([[-(tf2 - ti2)/(N*U)]*N] + In_moins + In_plus)

def funi(x):
    return np.dot(c, x)

def grad_funi(x):
    return c

def grad_c1(x):
    return A1.T

def grad_c2(x):
    return A2.T

def cont1(x1, x2):
    c1 = [(Qf1 - Qi1) - (tf1 - ti1)*sum(y for y in x1)/(N * U)]
    c2 = [-y for y in x1]
    c3 = [x1[i]* (1 - P_max/(x1[i] + x2[i])) for i in range(N)]
    return np.array(c1 + c2 + c3)

def cont2(x1, x2):
    c1 = [(Qf2 - Qi2) - (tf2 - ti2)*sum(y for y in x2)/(N * U)]
    c2 = [-y for y in x2]
    c3 = [x2[i]* (1 - P_max/(x1[i] + x2[i])) for i in range(N)]
    return np.array(c1 + c2 + c3)

def uzawa(xk1, xk2, grad_f1, grad_f2, grad_c1, grad_c2, c1, c2, epsilon=2.27):
    # 1 Initialisation
    lambdak = np.array([0]*(4*N + 2))
    rho = 0.02001
    lambdaswap = [42]*(4*N + 2)
    compteur = 0
    while np.linalg.norm(lambdaswap - lambdak) > epsilon:
        compteur += 1
        print(f"itération {compteur} avec {xk1} et {xk2} ")
        # 2 Décomposition
        lambdaswap = np.copy(lambdak)
        def decomp(xk, lk, grad_f, grad_c, alpha=1e-2, maxit=1e3, eps=1e-8):
            i = 0
            grad_l_xk = grad_f(xk) + np.dot(grad_c(xk), lk)
            while (i < maxit) and (np.linalg.norm(grad_l_xk > eps)):
                i += 1
                grad_l_xk = grad_f(xk) + np.dot(grad_c(xk), lk)
                xk = xk - alpha*grad_l_xk
            return xk
        xk1 = decomp(xk1, lambdak[0:2*N+1], grad_funi, grad_c1)
        xk2 = decomp(xk2, lambdak[2*N+1:], grad_funi, grad_c2)
        # 3 Coordination
        c1 = cont1(xk1, xk2)
        c2 = cont2(xk1, xk2)
        for i in range(0, 2*N+1):
            lambdak[i] = max(0, lambdak[i] + rho*c1[i])
            lambdak[i + 2*N+1] = max(0, lambdak[i + 2*N+1] + rho*c2[i])
        print(f"lambdak = {lambdak}")
    plt.plot(list(range(N)), xk1, color='red')
    plt.plot(list(range(N)), xk2, color='blue')
    plt.plot(list(range(N)), [Pmax]*N, color='green')
    return xk1, xk2