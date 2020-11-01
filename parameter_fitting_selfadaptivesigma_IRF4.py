''' Authors: Katinka den Nijs & Robin van den Berg '''

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
import csv


class IRF4(object):
    """ A class containing the parameters, equations and necessary functions for the standard Martinez model """

    def __init__(self, mu_r, sigma_r, CD40):
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.lambda_r = 1
        self.k_r = 1
        self.CD40 = CD40

    def equation(self, r):
        drdt = (self.mu_r + self.sigma_r * r ** 2 / (self.k_r ** 2 + r ** 2) + self.CD40 - \
                self.lambda_r * r)

        return drdt

    def calc_zeropoints(self):
        self.intersections = root(self.equation, [0., 6, 10], method='lm')
        return np.sort(self.intersections.x)

    def plot(self, name='test'):
        intersections = self.intersections.x
        r_list = np.arange(0, 12, 0.001)
        drdt = self.equation(r_list)

        # intersections = self.intersections.x
        fig = plt.figure()
        plt.title("{}: With intersections: {}".format(name, intersections))
        plt.plot(r_list, drdt, label='fit', color='green')
        plt.scatter(intersections, [0, 0, 0], marker='x')
        plt.scatter(CB_sep_data, np.zeros(len(CB_sep_data)), label='CB')
        plt.scatter(CC_sep_data, np.zeros(len(CC_sep_data)), label='CC')
        plt.scatter(PC_data, np.zeros(len(PC_data)), label='PC')
        plt.axhline(y=0, color='grey', linestyle='--')
        # plt.ylim(-0.05*10**-8, 0.09*10**(-8))
        # plt.xlim(3, 12)
        plt.xlabel('r', fontsize=14)
        plt.ylabel('drdt', fontsize=14)
        plt.legend(fontsize=14)
        fig.savefig("IRF4_mu_sigma_{}.png".format(name.capitalize()))
        plt.close(fig)


def fitness(ind):
    '''
    Calculates the fitness of an individual as the mse on the time series
    :param ind: an array containing the parameters we want to be fitting
    :return: the fitness of an individual, the lower the better tho
    '''

    ind = ind[:int(len(ind) / 2)]

    model_ind = IRF4(*ind)
    intersections = model_ind.calc_zeropoints()

    mu_r, sigma_r, CD40 = ind
    lambda_r = model_ind.lambda_r
    k_r = model_ind.k_r

    beta = (mu_r + CD40 + sigma_r) / (lambda_r * k_r)
    p = - sigma_r / (lambda_r * k_r) + beta

    if (beta ** 2 > 3) and (beta ** 3 + (beta ** 2 - 3) ** (3 / 2) - 9 * beta / 2 > - 27 / 2 * p) and \
            (beta ** 3 - (beta ** 2 - 3) ** (3 / 2) - 9 * beta / 2 < - 27 / 2 * p) and (beta > 0) and (p > 0) and \
            (mu_r > 0) and (sigma_r > 0) and (CD40 >= 0):

        return abs(sum((intersections[0] - GC_data) / GC_data)) / len(GC_data) + \
               abs(sum((intersections[2] - PC_data) / PC_data)) / len(PC_data)

    else:
        return 1000


def mutation(child):
    """
    Performs self-adaptive mutation
    """

    if np.random.random() < mutation_chance:
        num_of_sigmas = int(len(child) / 2)

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
    while child < nr_children - 2:
        parent1, idx1 = tournament_selection(population, k, fitnesses)
        parent2, idx2 = tournament_selection(population, k, fitnesses)

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
        indices_replaced[child + 1] = idx_replace2

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
                 abs(np.array([np.random.normal(0.1, 0.1, pop_size),
                               np.random.normal(2.6, 3, pop_size),
                               np.random.normal(0.5, 0.5, pop_size),
                               # ^ de waardes van de params (volgens Martinez), rest is sigmas
                               np.random.normal(0, 20, pop_size), np.random.normal(0, 20, pop_size),
                               np.random.normal(0, 20, pop_size)]).T)
    return population


def initializer(pop_size, d1, d2, d3, d4):
    global GC_data
    global PC_data
    global CB_sep_data
    global CC_sep_data
    global tau
    global mutation_chance

    GC_data = d1
    PC_data = d2
    CB_sep_data = d3
    CC_sep_data = d4

    mutation_chance = 0.25
    tau = 1 / math.sqrt(pop_size)


def run_evolutionary_algo(name_run, pop_size, num_variables, num_gen, tournament_size, GC_data, PC_data, CB_sep_data,
                          CC_sep_data):
    """
    Run evolutionary algorithm in parallel
    """

    len_ind = num_variables * 2

    population = init_pop(pop_size=pop_size, num_variables=len_ind)

    best_sol_current = None
    best_fit_current = 100000000000
    mu_list = np.zeros(num_gen * pop_size)
    sigma_list = np.zeros(num_gen * pop_size)
    CD40_list = np.zeros(num_gen * pop_size)
    fitness_list = np.zeros(num_gen * pop_size)

    with open("IRF4data_mu_sigma_{}.csv".format(name_run), "w") as file:
        writer = csv.DictWriter(file, delimiter=',', fieldnames=["generation", "fitness", 'mu_r', 'sigma_r', 'CD40'])
        writer.writeheader()

    # start evolutionary algorithm
    for gen in range(num_gen):
        start_time = time.time()
        pool_input = tuple(population)

        # run the different solutions in parallel
        pool = Pool(cpu_count(), initializer, (pop_size, GC_data, PC_data, CB_sep_data, CC_sep_data))
        fitnesses = pool.map(fitness, pool_input)
        pool.close()
        pool.join()

        mu_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 0]
        sigma_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 1]
        CD40_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 2]

        fitness_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = fitnesses

        best_fit_gen = min(fitnesses)

        if best_fit_gen < best_fit_current:
            best_sol_current = population[fitnesses.index(min(fitnesses))]
            best_fit_current = min(fitnesses)

        population = crossover(population, fitnesses, tournament_size, best_sol_current)

        pool_input = tuple(population)

        # run the different solutions in parallel
        pool = Pool(cpu_count(), initializer, (pop_size, GC_data, PC_data, CB_sep_data, CC_sep_data))
        population = pool.map(mutation, pool_input)
        pool.close()
        pool.join()

        print("Best fit and average fit at gen {}: \t {}, \t {}, time to run it {}".format(gen, best_fit_gen,
                                                                                           np.mean(fitnesses),
                                                                                           time.time() - start_time))

        with open("IRF4data_mu_sigma_{}.csv".format(name_run), "a") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([gen, best_fit_gen, *best_sol_current[:number_of_variables_to_fit]])

    results = pd.DataFrame(data=np.array([sigma_list, mu_list, CD40_list, fitness_list]).T,
                           columns=['sigma', 'mu', 'CD40', 'fitness'])
    results.to_csv('IRF4_last_pop_{}.csv'.format(name_run))

    return best_fit_current, best_sol_current


if __name__ == '__main__':
    # CHANGE NAME FOR NEW RUN!!!
    name_run = "musigma_WesData"

    crossover_chance = 0.25
    frac_to_replace = 0.3
    number_of_variables_to_fit = 3
    tournament_size = 20
    len_ind = number_of_variables_to_fit * 2  # times two for the sigmas
    pop_size = 10000
    num_gen = 30
    # tau = 1/math.sqrt(pop_size)   # is calculated in initializer function
    # mut_chance = 0.25  # needs to be changed in initializer function

    # affymetrix_df = pd.read_csv('matrinez_data.csv')  # change this in the initializer as well
    affymetrix_df = pd.read_csv('wesenhagen_data.csv')

    affymetrix_df = affymetrix_df.set_index('Sample')

    # affymetrix_df = affymetrix_df.divide(4)

    GC_data = np.append(affymetrix_df.loc['CB', 'IRF4'].values, affymetrix_df.loc['CC', 'IRF4'].values)
    PC_data = affymetrix_df.loc['PC', 'IRF4'].values
    CB_sep_data = affymetrix_df.loc['CB', 'IRF4'].values
    CC_sep_data = affymetrix_df.loc['CC', 'IRF4'].values

    best_fitness, best_ind = run_evolutionary_algo(name_run, pop_size, number_of_variables_to_fit, num_gen,
                                                   tournament_size,
                                                   GC_data, PC_data, CB_sep_data, CC_sep_data)

    best_solution = IRF4(*best_ind[:number_of_variables_to_fit])

    print('mu: {}, sigma: {}, CD40: {} '.format(best_solution.mu_r, best_solution.sigma_r,
                                                best_solution.CD40))
    print("De snijpunten met de x-as: ", best_solution.calc_zeropoints())
    print("The fitness of our solution: ",
          fitness([best_solution.mu_r, best_solution.sigma_r, best_solution.CD40, 0, 0, 0]))

    best_solution.plot(name_run)

    mu_r = 0.1
    sigma_r = 2.6
    CD40 = 0
    k_r = 1
    lambda_r = 1
    beta = mu_r + CD40 + sigma_r / (lambda_r * k_r)
    p = -sigma_r / (lambda_r * k_r) + beta
    sol_of_martinez = IRF4(mu_r, sigma_r, CD40)

    print('Martinez: mu: {}, sigma: {}, k: {}, lambda: {}, CD40: {} '.format(mu_r, sigma_r, k_r, lambda_r, CD40))
    print("Location of the roots: ", sol_of_martinez.calc_zeropoints())
    print("The fitness of the martinez solution: ", fitness([mu_r, sigma_r, CD40, 0, 0, 0]))

    sol_of_martinez.plot("Martinez_{}".format(name_run))
