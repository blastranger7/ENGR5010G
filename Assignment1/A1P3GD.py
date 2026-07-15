import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random

def project(x,y):

    return x,y

def main():
    x = 0.25
    y = 0.25
    alpha = 0.001
    num_iterations = 100
    mu = 0.5
    t = 0

    z = np.empty(num_iterations)
    for ii in range(num_iterations):
        zx = 2*x + 0.5*math.sin(0.2*t) 
        zy = 2*y + 0.5*math.cos(0.2*t) 
        x = x - alpha*zx
        y = y - alpha*zy
        z[ii] = (x**2 +y**2 +0.5*math.sin(0.2*t)*x + 0.5*math.cos(0.2*t)*y + np.random.normal(0,1) #Origional Loss Function
                 + mu*(max(0, 1-x-y)**3 + max(0, x**2 + y**2 -20)**2)) #Inequality Penalty Functions (Need Adjustment)

    print(f"Best fitness is {z[-1]} at ({x}, {y}))")
    plt.plot(z)
    plt.show()

if __name__ == "__main__":
    main()