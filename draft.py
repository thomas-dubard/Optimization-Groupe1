import matplotlib.pyplot as plt
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

def fun(x):
    return np.dot(c, x)

def grad_fun(x):
    return c

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

def contraintesactivesoQP(xk, alpha = 0.01, lamb=[0]*(2*N + 1), W=list(range(0, 2*N + 1))):
    """
    On implémente l'algorithme des contraintes actives QP.
    On est ici avec G=0, f(x)=x*c, c(x)=Ax-b
    où A=(-T/NU ... -T/NU) et b=(Qi - Qf)
         (     -In       )      (   0   )
         (      In       )      (  Pmax )
    """
    while not test_min(lamb):
        # (a) direction pk
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
            pk = [0]*N
            pk[y1], pk[y2] = c[y2], -c[y1]
        # (b)
        c_k = cont(xk)
        for i in range(2*N + 1):
            if lamb[i] != None:
                lamb[i] = min(0, lamb[i] + alpha*c_k[i])

    return xk