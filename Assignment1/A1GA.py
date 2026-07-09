import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random

def fitness(x,y):
    fit = 20 + (pow(x,2) + 10*math.sin(10*math.pi*x)) + (pow(y,2) + 10*math.sin(10*math.pi*y)) #fitness is just function value
    if (pow(x,2) + pow(y,2)) > 20 or (x+y) < 1:
        fit = fit + 100 #if outside of bounds penalize hard
    return fit

def initPop(pop_size):
    x = random.rand(10, size=(pop_size)) -5 #get random number between -5:5
    y = random.rand(10, size=(pop_size)) -5
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
        best_fit[jj] = y[opt_fit]

    return best_x, best_y, best_fit

def crossover(x,y):
    child_x = x
    child_y = y[np.size(y)/2+1:np.size(y)] 
    child_y = np.append(child_y, y[0:np.size(y)/2])
    return child_x, child_y

def mutate(x, y):
    child_x = x
    child_y = y
    fit = 0
    return child_x, child_y, fit

def main():
    #initialize variables
    pop_size = 100
    num_iterations = 100
    k = 5

    #initialize plotting matricies
    avg_fitness = np.empty(pop_size)
    best_fitness = np.empty(pop_size)

    x, y, fit = initPop(pop_size)
    for ii in range(num_iterations):
        best_x, best_y, best_fit = tournamentSelection(k, x, y, fit)
        child_x, child_y = crossover(best_x, best_y)
        child_x, child_y, child_fit = mutate(child_x, child_y)
        x = child_x
        y = child_y
        fit = child_fit
        avg_fitness[ii] = np.average(fit)
        best_fitness[ii] = np.min(fit)

    plt.plot(avg_fitness)
    plt.plot(best_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(["Average Fitness", "Best Fitness"])
    plt.show()