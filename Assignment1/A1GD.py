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

    z = np.empty(num_iterations)
    for ii in range(num_iterations):
        zx = 2*x + 20*math.pi*math.sin(2*math.pi*x)
        zy = 2*y + 20*math.pi*math.sin(2*math.pi*y)
        x = x - alpha*zx
        y = y - alpha*zy
        z[ii] = (20 + (x**2 - 10*math.cos(2*math.pi*x)) + (y**2 - 10*math.cos(2*math.pi*y)) #Origional Loss Function
                 + mu*(max(0, 1-x-y)**3 + max(0, x**2 + y**2 -20)**2)) #Inequality Penalty Functions (Need Adjustment)

    print(f"Best fitness is {z[-1]} at ({x}, {y}))")
    plt.plot(z)
    plt.show()

if __name__ == "__main__":
    main()