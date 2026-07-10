import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random

def fitness(x,y):
    #want to minimize function so lower fitness better
    fit = 20 + (x**2 - 10*math.cos(2*math.pi*x)) + (y**2 - 10*math.cos(2*math.pi*y)) #fitness is just function value
    if (x**2 + y**2) > 20 or (x+y) < 1:
        fit = fit + 100 #if outside of bounds penalize hard
    return fit

def initPop(pop_size):
    x = random.uniform(low=-5.0, high=5.0, size=pop_size) #get random float between -5:5
    y = random.uniform(low=-5.0, high=5.0, size=pop_size)
    fit = np.empty(pop_size)
    for ii in range(pop_size): #for each x,y pair get fitness
        fit[ii] = fitness(x[ii],y[ii])
    return x,y,fit

def tournamentSelection(k,x,y,fit):
    pop_size = np.size(fit)
    best_x = np.empty(pop_size) 
    best_y = np.empty(pop_size)
    best_fit = np.empty(pop_size)

    for jj in range(np.size(fit)):
        opt_fit = 1000
        opt_p = 0
        for ii in range(k): #select k random chromosomes and pick the best one
            p = random.randint(pop_size)
            if fit[p] < opt_fit:
                opt_fit = fit[p]
                opt_p = p
        best_x[jj] = x[opt_p]
        best_y[jj] = y[opt_p]
        best_fit[jj] = opt_fit

    return best_x, best_y, best_fit

def crossover(x,y):
    x = x
    y = np.flip(y) #reverse y for crossover
    return x, y

def mutate(x, y, rate):
    pop_size = np.size(x)
    fit = np.empty(pop_size)
    for ii in range(pop_size):
        if random.rand() < rate: #mutate with random noise
           x[ii] = x[ii] + random.rand()
           y[ii] = y[ii] + random.rand()
        fit[ii] = fitness(x[ii], y[ii]) 
    return x, y, fit

def gradDescent(x,y,num_iterations,alpha):
    pop_size = np.size(x)
    z = np.empty(pop_size)
    for ii in range(num_iterations):
        for jj in range(pop_size):
            zx = 2*x[jj] + 20*math.pi*math.sin(2*math.pi*x[jj])
            zy = 2*y[jj] + 20*math.pi*math.sin(2*math.pi*y[jj])
            x[jj] = x[jj] - alpha*zx
            y[jj] = y[jj] - alpha*zy
            z[jj] = fitness(x[jj], y[jj])
    return x, y, z

def main():
    #initialize variables
    pop_size = 500
    num_iterations = 100
    k = 10
    mutation_rate = 0.05

    gd_iterations = 5
    alpha = 0.001

    #initialize plotting matricies
    avg_fitness = np.empty(num_iterations)
    best_fitness = np.empty(num_iterations)

    x, y, fit = initPop(pop_size)
    for ii in range(num_iterations):
        avg_fitness[ii] = np.average(fit)
        best_fitness[ii] = np.min(fit)
        best_x, best_y, best_fit = tournamentSelection(k, x, y, fit)
        child_x, child_y = crossover(best_x, best_y)
        child_x, child_y, child_fit = mutate(child_x, child_y, mutation_rate)
        
        #Gradient Descent Hybridization
        child_x, child_y, child_fit = gradDescent(child_x, child_y, gd_iterations, alpha)
        
        x = child_x
        y = child_y
        fit = child_fit
        
    min_fit_pos = np.argmin(fit)  
    x_pos = x[min_fit_pos]
    y_pos = y[min_fit_pos]

    print(f"Best fitness is {np.min(fit):.4f} at ({x_pos:.4f}, {y_pos:.4f}))")
    plt.plot(avg_fitness)
    plt.plot(best_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(["Average Fitness", "Best Fitness"])
    plt.show()

if __name__ == "__main__":
    main()