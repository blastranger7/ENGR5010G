import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random

def fitness(x,y):
    fit = 20 + (x**2 + 10*math.cos(10*math.pi*x)) + (y**2 + 10*math.cos(10*math.pi*y)) #fitness is just function value
    if (x**2 + y**2) > 20 or (x+y) < 1:
        fit = fit + 100 #if outside of bounds penalize hard
    return fit

def initPop(pop_size):
    x = random.uniform(low=-5.0, high=5.0, size=pop_size) #get random number between -5:5
    y = random.uniform(low=-5.0, high=5.0, size=pop_size)
    fit = np.empty(pop_size)
    for ii in range(pop_size): #for each x,y pair get fitness
        fit[ii] = fitness(x[ii],y[ii])
    return x,y,fit

def tournamentSelection(k,x,y,fit):
    best_x = np.empty(np.size(fit)) 
    best_y = np.empty(np.size(fit))
    best_fit = np.empty(np.size(fit))

    for jj in range(np.size(fit)):
        opt_fit = 1000
        opt_p = 0
        for ii in range(k):
            p = random.randint(np.size(fit))
            if fit[p] < opt_fit:
                opt_fit = fit[p]
                opt_p = p
        best_x[jj] = x[opt_p]
        best_y[jj] = y[opt_p]
        best_fit[jj] = opt_fit

    return best_x, best_y, best_fit

def crossover(x,y):
    child_x = x
    child_y = np.flip(y)
    return child_x, child_y

def mutate(x, y, rate):
    fit = np.empty(np.size(x))
    for ii in range(np.size(x)):
        if random.rand() < rate: #randomly swap chromosomes with a certain probability
           x[ii] = x[ii] + random.rand()
           y[ii] = y[ii] + random.rand()
        fit[ii] = fitness(x[ii], y[ii])
    return x, y, fit

def main():
    #initialize variables
    pop_size = 500
    num_iterations = 100
    k = 10
    mutation_rate = 0.05

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
        x = child_x
        y = child_y
        fit = child_fit
        #print(fit)

    min_fit_pos = np.argmin(fit)  
    x_pos = x[min_fit_pos]
    y_pos = y[min_fit_pos]

    print(f"Best fitness is {np.min(fit):.4f} at ({x_pos}, {y_pos}))")
    plt.plot(avg_fitness)
    plt.plot(best_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(["Average Fitness", "Best Fitness"])
    plt.show()

if __name__ == "__main__":
    main()