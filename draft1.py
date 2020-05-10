import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time

"""
On se place directement entre t_0 et t_f.
Alors N correspond à l'échantillonage entre les deux.
"""


#c = np.array([1.0 for _ in range(N)]) # coût constant
#c = np.array([2.0*(math.cos(i) + 1.1) for i in range(N)]) # coût oscillant
c = np.array([(2.0*(i%3) + 1.0) for i in range(N)]) # coût "triangulaire"


N = 100


U = 230.0
Qf = 15.0
Qi = 0.0
t_f = 10.0
t_i = 0.0
P_max = 1.5*(Qf-Qi)/(t_f - t_i) * U
P_0 = [(Qf-Qi)/(t_f - t_i) * U]*N
W_0 = set([0])
In_plus = [[0 if x!=k else 1 for x in range(N)] for k in range(N)]
In_moins = [[0 if x!=k else -1 for x in range(N)] for k in range(N)]
A = np.array([[-(t_f - t_i)/(N*U)]*N] + In_moins + In_plus)
cst_pk = 0.1


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
        if l!=None and l <= 0.0:
            return False
    return True


def contraintesactivesOQP(c, xk=P_0, lambdak=np.array([0.0]*(2*N + 1)), W=W_0):
    debut = time.time()
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
    compteur = 0
    #compteur < (N-2)*(N-1)*N
    #cont(xk)[0]<=0
    while (not test_min(lambdak)) and all(y<=0.05 for y in cont(xk)):
        compteur += 1
        print(f"itération {compteur}, lambdak = {lambdak}")
        xswap = xk
        # (a) direction pk
        print("a")
        pk = [0]*N
        indice = []
        for i in W:
            if 1 < i <= N:
                indice.append(i - 1)
            else:
                indice.append(i - N - 1)
        indice = set(indice)
        if len(indice) < N - 3 and 0 in W:
            liste = [x for x in range(N) if not x in indice]
            i1 = random.randint(0, len(liste)-1)
            y1 = liste.pop(i1)
            i2 = random.randint(0, len(liste)-1)
            y2 = liste.pop(i2)
            i3 = random.randint(0, len(liste)-1)
            y3 = liste.pop(i3)
            sens = 2*random.randint(0, 1) - 1
            if c[y1] != c[y2]:
                pk[y1] = sens*(c[y2] - c[y3])/(c[y1] - c[y2])
                pk[y2] = sens*(c[y1] - c[y3])/(c[y2] - c[y1])
                pk[y3] = sens*1
            else:
                pk[y1] = 0
                pk[y2] = -sens
                pk[y3] = sens
            # (b) pk != 0
            # légitime car coût non nul à tout instant
            norm_pk = (sum(x**2 for x in pk))**0.5
            pk = [x/(cst_pk*norm_pk) for x in pk]
            print(f"b : pk={pk}, W={W}")
            swap = [1] + [None]*(2*N + 1)
            if not 0 in W and pk[y1] + pk[y2] + pk[y3] < 0:
                z = (Qi - Qf + (t_f - t_i)*sum(y for y in xk)/(N * U))
                swap[1] = (-z/(t_f - t_i)*(pk[y1] + pk[y2] + pk[y3])/(N * U))
            for i in range(1, N+1):
                if not i in W and pk[i-1] < 0:
                    swap[i+1] = (-xk[i-1]/pk[i-1])
                if not (i+N) in W and pk[i-1] > 0:
                    swap[i+1] = (xk[i-1]/pk[i-1])
            alphak = min(x for x in swap if x != None)
            xk = np.array([xk[i] + alphak * pk[i] for i in range(N)])
            if alphak < 1:
                j = 0
                while swap[j] != alphak:
                    j += 1
                W.add(j)
            print(f"xk={xk}, W={W}")
        elif len(indice) < N - 2 and not 0 in W:
            liste = [x for x in range(N) if not x in indice]
            i1 = random.randint(0, len(liste)-1)
            y1 = liste.pop(i1)
            i2 = random.randint(0, len(liste)-1)
            y2 = liste.pop(i2)
            sens = 2*random.randint(0, 1) - 1
            pk[y1] = sens*c[y2]
            pk[y2] = -sens*c[y1]
            # (b) pk != 0
            # légitime car coût non nul à tout instant
            norm_pk = (sum(x**2 for x in pk))**0.5
            pk = [x/(cst_pk*norm_pk) for x in pk]
            print(f"b : pk={pk}, W={W}")
            swap = [1] + [None]*(2*N + 1)
            if not 0 in W and pk[y1] + pk[y2] < 0:
                z = (Qi - Qf + (t_f - t_i)*sum(y for y in xk)/(N * U))
                swap[1] = (-z/(t_f - t_i)*(pk[y1] + pk[y2])/(N * U))
            for i in range(N+1):
                if not i in W and pk[i-1] < 0:
                    swap[i+1] = (-xk[i-1]/pk[i-1])
                if not (i+N) in W and pk[i-1] > 0:
                    swap[i+1] = (xk[i-1]/pk[i-1])
            alphak = min(x for x in swap if x != None)
            xk = np.array([xk[i] + alphak * pk[i] for i in range(N)])
            if alphak < 1:
                j = 0
                while swap[j] != alphak:
                    j += 1
                W.add(j)
            print(f"xk={xk}, W={W}")
        # (c) pk = 0
        def grad_lag(xk, lambdak):
            lambdak_full = np.array([0 if x==None else x for x in lambdak])
            return grad_fun(xk) + np.dot(A.T, lambdak_full)
        def step_c(xk, lk, grad_f, grad_l, c, alpha=1e-2, maxit=1e3, eps=1e-8):
            i = 0
            grad_l_xk = grad_l(xk, lk)
            while (i < maxit) and (np.linalg.norm(grad_l_xk > eps)):
                i += 1
                grad_l_xk = grad_l(xk, lk)
                xk = xk - alpha*grad_l_xk
                c_xk = c(xk)
                for j in range(len(lk)):
                    if j in W:
                        if lk[j] != None:
                            lk[j] = min(0, lk[j] + alpha*c_xk[j])
                        else:
                            lk[j] = min(0, alpha*c_xk[j])
                    else:
                        lk[j] = None
            return xk, lk
        if pk == [0]*N:
            print(f"c : pk={pk}")
            xk, lambdak = step_c(xk, lambdak, grad_fun, grad_lag, cont)
            lambdaswap = []
            for i in range(2*N + 1):
                if not i in W:
                    lambdaswap.append(None)
                else:
                    lambdaswap.append(lambdak[i])
            lambdak = np.array(lambdaswap)
            def swap_choose(lamdbak):
                i = 0
                while i < len(lambdak) and lambdak[i] == None:
                    i += 1
                return i, lambdak[i]
            swap = swap_choose(lambdak)
            for i in range(len(lambdak)):
                if lambdak[i] != None and lambdak[i] < swap[1]:
                    swap = i, lambdak[i]
            if swap[1] <= 0:
                xk = xswap
                W = set(x for x in W if x != swap[0])
            print(f"xk={xk}, W={W}")
            lambdaswap = []
            for i in range(2*N + 1):
                if not i in W:
                    lambdaswap.append(None)
                else:
                    lambdaswap.append(lambdak[i])
            lambdak = np.array(lambdaswap)
    fin = time.time()
    print(f"ça a pris {fin - debut} secondes pour N={N}, cont={cont(xk)} !")
    c = [x*(np.mean(xk)/np.mean(c)) for x in c]
    plt.plot(list(range(N)), xswap, color='red')
    plt.plot(list(range(N)), c, color='blue')
    plt.plot(list(range(N)), [P_max]*N, color='green')
    plt.plot(list(range(N)), P_0, color='yellow')
    plt.show()
    return xswap