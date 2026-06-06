import numpy as np

def f(x):
    return -1.3*x**2 + 3.3*x - 0.4

def tangente(x):
    return 1.22*(x-0.8) + 1.408

n = 20
noise = np.random.normal(0, 0.25, n)
X = np.linspace(0.1, 2.9, n)

Y_parabole = f(X)
Y_ligne = tangente(X)

noised_parab = Y_parabole + noise
noised_ligne = Y_ligne + noise

for i in range(n):
    print(X[i], noised_parab[i])
print("----------------")
for i in range(n):
    print(X[i], noised_ligne[i])