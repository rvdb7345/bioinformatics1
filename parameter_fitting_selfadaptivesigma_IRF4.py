import multiprocessing
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scipy.optimize import fsolve, root
import random
from numba import jit
import math
import time
from tqdm import tqdm
from mpl_toolkits import mplot3d



class IRF4(object):
    """ A class containing the parameters, equations and necessary functions for the standard Martinez model """

    def __init__(self, beta, p):
        self.beta = beta
        self.p = p

    def equation(self, y):
        # drdt = self.mu_r + self.sigma_r * r ** 2 / (self.k_r ** 2 + r ** 2) + CD40 - self.lambda_r * r

        F = (y**3 - self.beta * y**2 + y - self.p)

        return F

    def calc_zeropoints(self):
        self.intersections = root(self.equation, [0., 0.1, 10], method='lm')
        return np.sort(self.intersections.x)

    def plot(self, name='test'):
        intersections = self.intersections.x
        # intersections = np.sort(intersections)
        r_list = np.arange(0, 13, 0.001)
        drdt = self.equation(r_list)

        # intersections = self.intersections.x
        fig = plt.figure()
        plt.title("{}: With intersections: {}".format(name, intersections))
        # plt.plot(r_list + intersections[1], drdt - drdt[0])
        plt.plot(r_list, drdt, label='fit')
        plt.scatter(intersections, [0,0,0], marker='x')
        plt.scatter(inter1_data, np.zeros(len(inter1_data)), label='CC')
        plt.scatter(inter3_data, np.zeros(len(inter3_data)), label='CB')
        plt.scatter(inter2_data, np.zeros(len(inter2_data)), label='PC')
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.ylim(min(drdt), 5)
        plt.xlim(0,13)
        plt.xlabel('r', fontsize=14)
        plt.ylabel('drdt', fontsize=14)
        plt.legend(fontsize=14)
        fig.savefig("AffymetrixData{}.png".format(name.capitalize()))
        plt.close(fig)

def fitness(ind):
    '''
    Calculates the fitness of an individual as the mse on the time series
    :param ind: an array containing the parameters we want to be fitting
    :return: the fitness of an individual, the lower the better tho
    '''


    ind = ind[:int(len(ind)/2)]

    model_ind = IRF4(*ind)
    intersections = model_ind.calc_zeropoints()

    # instantiate an individual
    if (ind[0] ** 2 > 3) and (ind[0] ** 3 + (ind[0] ** 2 - 3) ** (3/2) - 9 * ind[0] / 2 > - 27/2 * ind[1]) and \
            (ind[0] ** 3 - (ind[0] ** 2 - 3) ** (3 / 2) - 9 * ind[0] / 2 < - 27/2 * ind[1]) and \
            (intersections[2] - intersections[0] > 0.2) and (ind[0] > 0) and (ind[1] > 0):

        # return abs(sum(intersections[0] - inter1_data)) + abs(sum(intersections[2] - inter2_data)),
        # print(np.linalg.norm(np.full((1, len(inter1_data)), intersections[0])))

        a = np.empty(len(inter1_data))
        a.fill(intersections[0])


        b = np.empty(len(inter2_data))
        b.fill(intersections[2])

        # print(len(a), len(b), len(inter1_data), len(inter2_data))
        # return abs(np.linalg.norm((a, inter1_data))) + \
        #        abs(np.linalg.norm((b, inter2_data))),
        return abs(sum(intersections[0] - inter1_data))/len(inter1_data) + \
               abs(sum(intersections[2] - inter2_data))/len(inter2_data)

    else:
        return 20

# @jit(nopython=True)
def mutation(child):
    """
    Performs self-adaptive mutation
    """

    if np.random.random() < mutation_chance:
        num_of_sigmas = int(len(child)/2)

        # get new mutation step size for all parameters
        child[num_of_sigmas:] = child[num_of_sigmas:] * np.exp(tau * np.random.normal(0, 1))

        # mutate child
        child[:num_of_sigmas] = abs(child[:num_of_sigmas] + np.random.normal(num_of_sigmas) * child[num_of_sigmas:])

    return child

def crossover(population, fitnesses, k, best_sol):
    """
    Performs uniform crossover with tournament selection
    """
    nr_children = int(len(population) * frac_to_replace)
    # print("The number of childern that will be replaced", nr_children)
    child = 1

    indices_replaced = np.zeros(nr_children)

    # start making offspring (two for each couple of parents)
    while child < nr_children-2:
        parent1, idx1 = tournament_selection(population, k, fitnesses)
        parent2, idx2 = tournament_selection(population, k, fitnesses)
        #
        # print("parent one with idx and fitness: ", idx1, fitnesses[idx1])
        # print("parent two with idx and fitness: ", idx2, fitnesses[idx2])

        # determine weights
        weights = np.zeros(len_ind)
        weights[:len_ind] = np.random.choice(np.array([0, 1]), size=len_ind)
        np.random.shuffle(weights)

        # perform crossover
        offspring1 = weights * parent1 + (1 - weights) * parent2
        offspring2 = (1 - weights) * parent1 + weights * parent2

        ind_to_be_replaced_1, idx_replace1 = tournament_selection_worst(population, k, fitnesses, indices_replaced)
        ind_to_be_replaced_2, idx_replace2 = tournament_selection_worst(population, k, fitnesses, indices_replaced)

        indices_replaced[child] = idx_replace1
        indices_replaced[child+1] = idx_replace2

        population[idx_replace1] = abs(offspring1)
        population[idx_replace2] = abs(offspring2)
        child += 2

    population[0] = best_sol

    return population

def tournament_selection_worst(sols, k, fitnesses, indices_replaced):
    '''
        Selects the best individuals out of k individuals
    '''

    best = -1

    for i in range(k):
        idx = np.random.randint(0, len(sols))

        if (best == -1 or fitnesses[idx] > fitnesses[best]) and not (idx in indices_replaced):
            best = idx

    return sols[best], best

def tournament_selection(sols, k, fitnesses):
    '''
        Selects the best individuals out of k individuals
    '''

    best = -1

    for i in range(k):
        idx = np.random.randint(0, len(sols))

        if best == -1 or fitnesses[idx] < fitnesses[best]:
            best = idx

    return sols[best], best

def init_pop(pop_size, num_variables):
    population = np.ones((pop_size, num_variables)) * \
        abs(np.array([np.random.normal(10, 2, pop_size), np.random.normal(0.05, 0.025, pop_size),
                      np.random.normal(1, 1, pop_size), np.random.normal(0.05, 0.025, pop_size)]).T)
    return population

def initializer(d1, d2, d3):
    global inter1_data
    global inter2_data
    global inter3_data
    global tau
    global mutation_chance

    inter1_data = d1
    inter3_data = d3
    inter2_data = d2

    mutation_chance = 0.25
    tau = 0.90

def run_evolutionary_algo(pop_size, num_variables, num_gen, tournament_size, inter1_data, inter2_data, inter3_data):
    """
    Run evolutionary algorithm in parallel
    """

    len_ind = num_variables * 2

    population = init_pop(pop_size=pop_size, num_variables=len_ind)
    # pool_input = [tuple(ind) for ind in population]


    best_sol_current = None
    best_fit_current = 100000000000
    betas_list = np.zeros(num_gen*pop_size)
    p_list = np.zeros(num_gen * pop_size)
    fitness_list = np.zeros(num_gen * pop_size)


    # start evolutionary algorithm
    for gen in range(num_gen):
        start_time = time.time()
        pool_input = tuple(population)

        # run the different solutions in parallel
        pool = Pool(cpu_count(), initializer, (inter1_data, inter2_data, inter3_data))
        fitnesses = pool.map(fitness, pool_input)
        pool.close()
        pool.join()


        betas_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 0]
        p_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 1]
        fitness_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = fitnesses

        best_fit_gen = min(fitnesses)

        if best_fit_gen < best_fit_current:
            best_sol_current = population[fitnesses.index(min(fitnesses))]
            best_fit_current = min(fitnesses)

        population = crossover(population, fitnesses, tournament_size, best_sol_current)

        pool_input = tuple(population)

        # run the different solutions in parallel
        pool = Pool(cpu_count(), initializer, (inter1_data, inter2_data, inter3_data))
        population = pool.map(mutation, pool_input)
        pool.close()
        pool.join()


        print("Best fit and average fit at gen {}: \t {}, \t {}, time to run it {}".format(gen, best_fit_gen,
                                                                                           np.mean(fitnesses),
                                                                                           time.time()-start_time))


    results = pd.DataFrame(data=np.array([betas_list, p_list, fitness_list]).T, columns=['beta', 'p', 'fitness'])
    results.to_csv('IRF4_fitting_individuals_betap.csv')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter3D(betas_list, p_list, zs=fitness_list, c=fitness_list,  cmap='hot', marker='.')
    ax.set_xlabel('beta', fontweight='bold')
    ax.set_ylabel('p', fontweight='bold')
    ax.set_zlabel('fitness', fontweight='bold')
    ax.set_xlim([0,12])
    ax.set_ylim([0,0.1])
    ax.set_zlim([0,12])
    fig.colorbar(scat, ax=ax)

    plt.title("Fitnesses in Solution Space")
    plt.show()

    return best_fit_current, best_sol_current



if __name__ == '__main__':
    # tau = 0.99   # needs to be changed in initializer function
    # mutation_chance = 0.25  # needs to be changed in initializer function
    crossover_chance = 0.25
    frac_to_replace = 0.3
    number_of_variables_to_fit = 2
    tournament_size = 20
    len_ind = number_of_variables_to_fit * 2  # times two for the sigmas
    pop_size = 100000
    num_gen = 20


    # #
    # # print("our best individual has a fitness of: ", best_fitness)
    # # print('With the following genetic material: ', best_ind)
    # #
    # affymetrix_df = pd.read_csv('matrinez_data.csv')  # change this in the initializer as well
    affymetrix_df = pd.read_csv('wesenhagen_data.csv')
    #
    # # # affymetrix_df = affymetrix_df.groupby('Sample').agg({'PRDM1':'mean','BCL6':'mean','IRF4':'mean'})
    affymetrix_df = affymetrix_df.set_index('Sample')
    inter1_data = np.append(affymetrix_df.loc['CB', 'IRF4'].values, affymetrix_df.loc['CC', 'IRF4'].values)
    # # inter1_data = affymetrix_df.loc['CC', 'IRF4'].values / 4
    inter3_data = affymetrix_df.loc['CB', 'IRF4'].values
    inter2_data = affymetrix_df.loc['PC', 'IRF4'].values

    best_fitness, best_ind = run_evolutionary_algo(pop_size, number_of_variables_to_fit, num_gen, tournament_size,
                                                   inter1_data, inter2_data, inter3_data)

    #
    best_solution = IRF4(*best_ind[:number_of_variables_to_fit])
    # best_solution = IRF4(9.813971426023775, 0.025607569275613286)

    print('beta, p: ', best_solution.beta, best_solution.p)
    print("De snijpunten met de x-as: ", best_solution.calc_zeropoints())
    print("The fitness of our solution: ", fitness([best_solution.beta, best_solution.p, 0, 0]))

    best_solution.plot('Ours')

    mu_r = 0.1
    sigma_r = 2.6
    CD40 = 0
    lambda_r = 1
    k_r = 1
    beta = mu_r + CD40 + sigma_r / (lambda_r * k_r)
    p = -sigma_r / (lambda_r * k_r) + beta
    # beta = 12.76144062324778
    # p = 0.004582577456956276
    sol_of_martinez = IRF4(beta, p)

    print('Beta and p of martinez: {}, {}'.format(beta, p))
    print('sigma_r and mu_r of martinez: {}, {}'.format(sigma_r, mu_r))
    print("Location of the roots: ", sol_of_martinez.calc_zeropoints())
    print("The fitness of the martinez solution: ", fitness([beta, p, 0, 0]))

    sol_of_martinez.plot('Martinez')


