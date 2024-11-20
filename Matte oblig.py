import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

alpha = 1.  
beta = 1.
delta = 1.
gamma = 1.
x0 = 4.  
y0 = 2.  

def derivative(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

def Euler(func, X0, t, alpha, beta, delta, gamma):
   
    #Euler løsning for ODEs.

    dt = t[1] - t[0]  
    nt = len(t)       
    X = np.zeros([nt, len(X0)])  
    X[0] = X0                    
    for i in range(nt - 1):
        X[i + 1] = X[i] + func(X[i], t[i], alpha, beta, delta, gamma) * dt
    return X

Nt = 1000
tmax = 30.
t = np.linspace(0., tmax, Nt)
X0 = [x0, y0]

Xe = Euler(derivative, X0, t, alpha, beta, delta, gamma)

res = integrate.odeint(derivative, X0, t, args=(alpha, beta, delta, gamma))
x, y = res.T

plt.figure()
plt.title("Euler Metode - Florida man vs. Alligator")
plt.plot(t, Xe[:, 0], 'b', label='Florida man')
plt.plot(t, Xe[:, 1], 'r', label='Alligator')
plt.grid()
plt.xlabel("Time, $t$ [s]")
plt.ylabel('Population')
plt.ylim([0., max(np.max(Xe[:, 0]), np.max(Xe[:, 1])) + 1])
plt.legend(loc="best")
plt.show()

plt.figure()
plt.title("odeint løsning - Florida man vs. Alligator")
plt.plot(t, x, 'b', label='Florida man')
plt.plot(t, y, 'r', label='Alligator')
plt.grid()
plt.xlabel("Time, $t$ [s]")
plt.ylabel('Population')
plt.ylim([0., max(np.max(x), np.max(y)) + 1])
plt.legend(loc="best")
plt.show()

plt.figure()
plt.plot(Xe[:, 0], Xe[:, 1], "-")
plt.xlabel("Florida man")
plt.ylabel("Alligator")
plt.grid()
plt.title("Fase plan - Florida man vs. Alligator (Euler)")
plt.show()

plt.figure()
IC = np.linspace(1.0, 6.0, 21) 
for florida_man in IC:
    X0 = [florida_man, 1.0]
    Xs = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma))
    plt.plot(Xs[:,0], Xs[:,1], "-", label = "$x_0 =$"+str(X0[0]))
plt.xlabel("Florida man")
plt.ylabel("Alligator")
plt.legend()
plt.title("Fase plan - Florida man vs. Alligator (odeint)")
plt.show() 