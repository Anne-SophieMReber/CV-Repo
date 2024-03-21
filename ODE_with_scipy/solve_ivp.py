import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#set variables as in problem
m = 0.1 #kg
k = 2 #kg/s^2
alpha = 4 #kg/s^2

#set initial conditions, postition is 0 everywhere, momentum is 0 everywhere but [0] = 0.1
phi_int = np.zeros((15))
x_pos = np.linspace(1,15,15)
momentum_int = np.zeros((15))
momentum_int[0] = 0.1
z_int =np.concatenate((momentum_int,phi_int)) #initial conditions flattened for solve_ivp function

def oscilatting_system(t, z):
    #retruns the derivative functions for phi postition and for momentum

    momentum = z[:15] #taking out momentum from z
    phi = np.zeros(17) #adding dummy values phi 0 and phi N+1 to calculate momentum in for loop
    phi[1:16] = z[15::] #taking out phi values from z

    phi[0] = z[15] #taking out phi values from z
    phi[-1] = z[-1] #taking out phi values from z
    
    #Calculate time derivative of momentum
    mom_update = np.zeros(15)
    for i in range(len(momentum)):
        mom_update[i] = -1*(k*phi[i+1]+alpha*(2*phi[i+1]-phi[i]-phi[i+2])) #version without counting twice

    #Calculate time derivative of postition in phi direction
    phi_update = momentum/m
    return np.concatenate((mom_update,phi_update)) #again concatenate to retain same shape as z



#solving with dense time output
sol = solve_ivp(oscilatting_system, [0, 10], y0= z_int, dense_output = True)#, first_step=dt)
t = np.linspace(0,10,10000)
z = sol.sol(t)
momentum_evolution = z[:15,:]
phi_evolution = z[15::,:]


plt.plot(x_pos, phi_evolution[::,0:600:150], "o-")
plt.title("Between 0 and 0.6 seconds")
plt.grid()
plt.show()

plt.plot(x_pos, phi_evolution[::,1400:1800:100], "o-")
plt.title("Between 1.4 and 1.8 seconds")
plt.grid()
plt.show()

plt.plot(x_pos, phi_evolution[::,5000:5400:100], "o-")
plt.grid()
plt.show()

plt.plot(x_pos, phi_evolution[::,9600:10000:200], "o-")
plt.grid()
plt.show()
