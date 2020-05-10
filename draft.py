#import matplotlib.pyplot as plt
import numpy as np
"""
On se place directement entre t_0 et t_f.
Alors N correspond à l'échantillonage entre les deux.
"""
N = 100
U = 230.0
c = np.array([1.0 for _ in range(N)]) # coût constant
Qf = 500.0
Qi = 0.0
t_f = 10.0
t_i = 0.0
P_max = 500.0
P_0 = [(Qf-Qi)/(t_f - t_i) * U/N]*N
W_0 = list(range(0, 2*N + 1))
In_plus = [[0 if x!=k else 1 for x in range(N)] for k in range(N)]
In_moins = [[0 if x!=k else -1 for x in range(N)] for k in range(N)]
A = np.array([[-(t_f - t_i)/(N*U)]*N] + In_moins + In_plus)

def fun(x):
    return np.dot(c, x)

def grad_fun(x):
    return c

def grad_c(x):
    return A.T

def cont(x):
    c1 = [(Qf - Qi) - (t_f - t_i)*sum(y for y in x)/(N * U)]
    c2 = [-y for y in x]
    c3 = [y - P_max for y in x]
    return np.array(c1 + c2 + c3)

def test_min(lamb):
    for l in lamb:
        if l == None or l <= 0:
            return False
    return True

def contraintesactivesOQP(xk=P_0, lambdak=[0]*(2*N + 1), W=W_0):
    """
    On implémente l'algorithme des contraintes actives QP.
    On est ici avec G=0, f(x)=x*c, c(x)=Ax-b
    où A=(-T/NU ... -T/NU) et b=(Qi - Qf)
         (     -In       )      (   0   )
         (      In       )      (  Pmax )
    Dans l'étape (a) on cherche une direction pour la recherche.
    On la choisit aléatoirement.
    Elle est telle que A*p=0, ce qui est garanti par notre choix simple.
    Elle minimise c*p.
    """
    while not test_min(lambdak):
        xswap = xk
        # (a) direction pk
        pk = [0]*N
        indice = []
        for i in W:
            if 0 < i <= N:
                indice.append(i)
            else:
                indice.append(i - N)
        indice = set(indice)
        if len(indice) < N - 1:
            liste = [x for x in range(N) if not x in indice]
            i1 = random.randint(len(liste))
            y1 = liste.pop(i1)
            i2 = random.randint(len(liste))
            y2 = liste.pop(i2)
            pk[y1], pk[y2] = c[y2], -c[y1]
            # (b) pk != 0
            # légitime car coût non nul à tout instant
            swap = [1] + [None]*(2*N + 1)
            if not 0 in W and pk[y1] + pk[y2] < 0:
                z = (Qi - Qf + (t_f - t_i)*sum(y for y in xk)/(N * U))
                swap[1] = (-z/(t_f - t_i)*(pk[y1] + pk[y2])/(N * U))
            for i in range(N+1):
                if not i in W and pk[i-1] < 0:
                    swap[i+1] = (-xk[i-1]/pk[i-1])
                if not (i+N) in W and pk[i+N-1] > 0:
                    swap[i+1] = (xk[i-1]/pk[i-1])
            alphak = min(x for x in swap if x != None)
            xk = xk + alphak * pk
            if alphak < 1:
                j = 0
                while swap[j] != alphak:
                    j += 1
                W.append(j)
        # (c) pk = 0
        def grad_lag(xk, lambdak):
            return grad_fun(xk) + np.dot(A.T, lambdak)
        def step_c(xk, lk, grad_f, grad_l, c, alpha=1e-2, maxit=1e3, eps=1e-8):
            i = 0
            grad_l_xk = grad_l(xk, lk)
            while (i < maxit) and (grad_l_xk > eps):
                grad_l_xk = grad_l(xk, lk)
                xk = xk - alpha*grad_l_xk
                for x in lk:
                    x = min(O, x + alpha*c(xk))
            return xk, lk
        if pk == [O]*N:
            xk, lambdak = step_c(xk, lambdak, grad_fun, grad_lag, cont)
            for i in range(2*N + 1):
                if not i in W:
                    lambdak[i] = None
            swap = 0, lambdak[0]
            for i in range(len(lambdak)):
                if lambdak[i] < swap[1]:
                    swap = i, lambdak[i]
            if swap[1] <= 0:
                xk = xswap
                W = [x for x in W if x != swap[0]]
    return xk