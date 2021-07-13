import math

import numpy
import numpy as np
import scipy
from scipy.stats import hypergeom
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def hypergeometric(A, B, n):
    N = A + B
    p = A / N
    j = 0
    X = 0
    if n >= N:
        return "date incorecte!"
    while j != n:
        U = np.random.uniform(low=0.0, high=1.0, size=None)
        if U < p:
            X = X + 1
            S = 1
        else:
            S = 0
        N = N - 1
        A = A - S
        p = A / N
        n = n - 1
    return X


def hypergeometric2(A, B, n):
    N = A + B
    p = A / N
    X = 0
    if n >= N or A<0 or B<0:
        return "date incorecte!"
    for i in range(n, 0, -1):
        U = np.random.uniform(low=0.0, high=1.0, size=None)
        if U < p:
            X = X + 1
            S = 1
        else:
            S = 0
        N = N - 1
        A = A - S
        p = A / N
    return X

def beta(a, b):
    while True:
        U1 = np.random.uniform(low=0.0, high=1.0, size=None)
        U2 = np.random.uniform(low=0.0, high=1.0, size=None)
        V = U1**(1/a)
        T = U2**(1/(b-1))
        if (V + T < 1):
            break

    X = V / (V + T)
    return X

def Exp():
    N = 0
    while True:
        U0 = np.random.uniform(low=0.0, high=1.0, size=None)
        U1 = np.random.uniform(low=0.0, high=1.0, size=None)
        Ux = U0
        K = 1
        while U0 < U1:
            K = K + 1
            U0 = U1
            U1 = np.random.uniform(low=0.0, high=1.0, size=None)
        if K % 2 ==1:
            X = N + Ux
            return X
        else:
            N=N+1
            continue

def X1(v):
    while True:
        U = np.random.uniform(low=0.0, high=1.0, size=None)
        Z0 = U**(1/v)
        Z1 = np.random.uniform(low=0.0, high=1.0, size=None)
        K = 1
        Zx = Z0
        while Z0 >= Z1:
            Z0 = Z1
            Z1 = np.random.uniform(low=0.0, high=1.0, size=None)
            K = K + 1
        if K % 2 == 1:
            return Zx
        else:
            continue

def X2(v):
    while True:
        U = np.random.uniform(low=0.0, high=1.0, size=None)
        Z = U ** ((-1) * ( 1/ (1-v) ))
        X0 = Exp()
        Y = X0 + 1
        if (Y <= Z):
            continue
        else:
            return Y

def gama (x):
    if (x < 1):
        p1 = x
        p2 = 1 - p1
        U = np.random.uniform(low=0.0, high=1.0, size=None)
        if U <= p1:
            x1 = X1(x)
            return x1
        else:
            x2 = X2(x)
            return x2

    if (x > 1):
        v = x
        b = v - 1
        c = v + b
        s = np.sqrt(2*v-1)
        while True:
            U = np.random.uniform(low=0.0, high=1.0, size=None)
            T = s * np.tan(math.pi*(U-0.5))
            Y = b + T
            if Y > 0:
                U1 = np.random.uniform(low=0.0, high=1.0, size=None)
                if U1 <= math.e ** (b * math.log(Y/b)-T + math.log(1 + T**2/c)):
                    return Y
                else:
                    continue
            else:
                continue

def beta2(a, b):
    x1 = gama(a); #Gamma(0,1,a)
    x2 = gama(b);  #Gamma(0,1,b)
    return x1/(x1 + x2)


if __name__ == '__main__':
    print("\nVariablila Beta\n")
    nrSim = int(input("numar simulari (recomandat >1000)= "))
    a = 0.75
    b = 4
    print('\033[92m'+"media teoretica: " + str(1 / (1 + b / a)) + '\033[0m')
    print('\033[92m'+"dispersia teoretica: " + str((a*b) / ((a+b)**2 * (a+b+1)) ) +'\033[0m' )

    print("Metoda 1 (teorema): ")
    sum = 0
    sumDisp = 0
    sum2 = 0
    sumDisp2 = 0
    for i in range(0,nrSim):
        B = beta(0.75, 4)
        B2 = beta2(0.75, 4)
        sum = sum + B
        sumDisp = sumDisp + B**2
        sum2 = sum2 + B2
        sumDisp2 = sumDisp2 + B2**2
    print("media empirica: " + str(sum/nrSim))
    print("dispersia empirica: " + str(sumDisp / nrSim - (sum / nrSim) ** 2))
    print("Metoda 2 (din 2 variabile Gamma): ")
    print("media empirica: " + str(sum2 / nrSim))
    print("dispersia empirica: " + str(sumDisp2 / nrSim - (sum2 / nrSim) ** 2))
