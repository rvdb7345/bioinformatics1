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


class full_model(object):
    """ A class containing the parameters, equations and necessary functions for the standard Martinez model """

    def __init__(self, CD40, mu_r, sigma_r, k_r, l_r, mu_b, sigma_b, k_b, l_b, mu_p, sigma_p, k_p, l_p):
        self.CD40 = CD40

        self.mu_r = mu_r
        self.sigma_r = sigma_r

        self.mu_b = mu_b
        self.sigma_b = sigma_b

        self.mu_p = mu_p
        self.sigma_p = sigma_p

        # Dissociation constant
        self.k_p = k_p
        self.k_b = k_b
        self.k_r = k_r
        # Degradation rate
        self.l_p = l_p
        self.l_b = l_b
        self.l_r = l_r

    def equation_irf(self, r):
        drdt = (self.mu_r + self.sigma_r * r ** 2 / (self.k_r ** 2 + r ** 2) + self.CD40 - \
                self.l_r * r)

        return drdt

    def equation_bcl_GC(self, b):
        dbdt = self.mu_b + (self.sigma_b * self.k_p ** 2) / (self.k_p ** 2 + np.mean(inter1_data_BLIMP) ** 2) * (
                self.k_b ** 2) / (self.k_b ** 2 + b ** 2) * (self.k_r ** 2) / \
               (self.k_r ** 2 + np.mean(inter1_data_IRF) ** 2) - (self.l_p) * b

        return dbdt

    def equation_bcl_PC(self, b):
        dbdt = self.mu_b + (self.sigma_b * self.k_p ** 2) / (self.k_p ** 2 + np.mean(inter2_data_BLIMP) ** 2) * (self.k_b ** 2) / \
               (self.k_b ** 2 + b ** 2) * (self.k_r ** 2) / (self.k_r ** 2 + np.mean(inter2_data_IRF) ** 2) - (self.l_p) * b

        return dbdt

    def equation_blimp_GC(self, p):
        dpdt = self.mu_p + (self.sigma_p * self.k_b ** 2) / (self.k_b ** 2 + np.mean(inter1_data_BCL) ** 2) + \
               (self.sigma_p * np.mean(inter1_data_IRF) ** 2) / (self.k_r ** 2 + np.mean(inter1_data_IRF) ** 2) - self.l_p * p

        return dpdt

    def equation_blimp_PC(self, p):
        dpdt = self.mu_p + (self.sigma_p * self.k_b ** 2) / (self.k_b ** 2 + np.mean(inter2_data_BCL) ** 2) + \
               (self.sigma_p * np.mean(inter2_data_IRF) ** 2) / (self.k_r ** 2 + np.mean(inter2_data_IRF) ** 2) - self.l_p * p

        return dpdt

    def calc_zeropoints(self):
        self.intersections_irf = root(self.equation_irf, [0., 8, 200], method='lm')
        self.intersections_bcl_GC = root(self.equation_bcl_GC, np.mean(inter1_data_BCL), method='lm')
        self.intersections_bcl_PC = root(self.equation_bcl_PC, np.mean(inter2_data_BCL), method='lm')
        self.intersections_blimp_GC = root(self.equation_blimp_GC, np.mean(inter1_data_BLIMP), method='lm')
        self.intersections_blimp_PC = root(self.equation_blimp_PC, np.mean(inter2_data_BLIMP), method='lm')

        return (self.intersections_irf.x, [self.intersections_bcl_GC.x, \
               self.intersections_bcl_PC.x], [self.intersections_blimp_GC.x, \
               self.intersections_blimp_PC.x])

    def plot(self, name='test', gen=0):
        # intersections = np.sort(intersections)
        r_list = np.arange(0, 12, 0.001)
        drdt = self.equation_irf(r_list)
        dbdt_GC = self.equation_bcl_GC(r_list)
        dbdt_PC = self.equation_bcl_PC(r_list)
        dpdt_GC = self.equation_blimp_GC(r_list)
        dpdt_PC = self.equation_blimp_PC(r_list)



        # intersections = self.intersections.x
        fig = plt.figure()
        plt.title("{}: With intersections: {}".format(name, self.intersections_irf.x))
        # plt.plot(r_list + intersections[1], drdt - drdt[0])
        plt.plot(r_list, drdt, label='fit')
        plt.scatter(self.intersections_irf.x, [0, 0, 0], marker='x')
        plt.scatter(inter1_data_IRF, np.zeros(len(inter1_data_IRF)), label='GC')
        plt.scatter(inter2_data_IRF, np.zeros(len(inter2_data_IRF)), label='PC')
        plt.axhline(y=0, color='grey', linestyle='--')
        # plt.ylim(-0.05 * 10 ** -8, 0.09 * 10 ** (-8))
        # plt.xlim(3, 12)
        plt.xlabel('r', fontsize=14)
        plt.ylabel('drdt', fontsize=14)
        plt.legend(fontsize=14)
        fig.savefig("AffymetrixData{}_IRF4_kl_at{}.png".format(name.capitalize(), gen))
        plt.close(fig)

        fig = plt.figure()
        plt.title("{}: With intersections: {}".format(name, [self.intersections_bcl_GC.x, self.intersections_bcl_PC.x]))
        # plt.plot(r_list + intersections[1], drdt - drdt[0])
        plt.plot(r_list, dbdt_PC, label='fit PC')
        plt.plot(r_list, dbdt_GC, label='fit GC')
        plt.scatter([self.intersections_bcl_GC.x, self.intersections_bcl_PC.x], [0, 0], marker='x')
        plt.scatter(inter1_data_BCL, np.zeros(len(inter1_data_BCL)), label='GC')
        plt.scatter(inter2_data_BCL, np.zeros(len(inter2_data_BCL)), label='PC')
        plt.axhline(y=0, color='grey', linestyle='--')
        # plt.ylim(-0.05 * 10 ** -8, 0.09 * 10 ** (-8))
        # plt.xlim(3, 12)
        plt.xlabel('b', fontsize=14)
        plt.ylabel('dbdt', fontsize=14)
        plt.legend(fontsize=14)
        fig.savefig("AffymetrixData{}_BCL_kl_at{}.png".format(name.capitalize(), gen))
        plt.close(fig)

        fig = plt.figure()
        plt.title("{}: With intersections: {}".format(name, [self.intersections_blimp_GC.x, self.intersections_blimp_PC.x]))
        # plt.plot(r_list + intersections[1], drdt - drdt[0])
        plt.plot(r_list, dpdt_PC, label='fit PC')
        plt.plot(r_list, dpdt_GC, label='fit GC')
        plt.scatter([self.intersections_blimp_GC.x, self.intersections_blimp_PC.x], [0, 0], marker='x')
        plt.scatter(inter1_data_BLIMP, np.zeros(len(inter1_data_BLIMP)), label='GC')
        plt.scatter(inter2_data_BLIMP, np.zeros(len(inter2_data_BLIMP)), label='PC')
        plt.axhline(y=0, color='grey', linestyle='--')
        # plt.ylim(-0.05 * 10 ** -8, 0.09 * 10 ** (-8))
        # plt.xlim(3, 12)
        plt.xlabel('p', fontsize=14)
        plt.ylabel('dpdt', fontsize=14)
        plt.legend(fontsize=14)
        fig.savefig("AffymetrixData{}_BLIMP_kl_at{}.png".format(name.capitalize(), gen))
        plt.close(fig)

def fitness(ind):
    '''
    Calculates the fitness of an individual as the mse on the time series
    :param ind: an array containing the parameters we want to be fitting
    :return: the fitness of an individual, the lower the better tho
    '''

    ind = ind[:int(len(ind) / 2)]

    model_ind = full_model(*ind)
    intersections = model_ind.calc_zeropoints()

    beta = (model_ind.mu_r + model_ind.CD40 + model_ind.sigma_r) / (model_ind.l_r * model_ind.k_r)
    p = - model_ind.sigma_r / (model_ind.l_r * model_ind.k_r) + beta

    # print((beta ** 2 > 3) , (beta ** 3 + (beta ** 2 - 3) ** (3 / 2) - 9 * beta / 2 > - 27 / 2 * p) , \
    #         (beta ** 3 - (beta ** 2 - 3) ** (3 / 2) - 9 * beta / 2 < - 27 / 2 * p) , (beta > 0) , (p > 0))

    # instantiate an individual
    if (beta ** 2 > 3) and (beta ** 3 + (beta ** 2 - 3) ** (3 / 2) - 9 * beta / 2 > - 27 / 2 * p) and \
            (beta ** 3 - (beta ** 2 - 3) ** (3 / 2) - 9 * beta / 2 < - 27 / 2 * p) and (beta > 0) and (p > 0) \
        and (ind[0] > 0) and (ind[1] > 0) and (ind[2] > 0) and (ind[3] > 0) and (ind[4] > 0):

        # return abs(sum(intersections[0] - inter1_data)) + abs(sum(intersections[2] - inter2_data)),
        # print(np.linalg.norm(np.full((1, len(inter1_data)), intersections[0])))

        # a = np.empty(len(inter1_data))
        # a.fill(intersections[0])
        #
        # b = np.empty(len(inter2_data))
        # b.fill(intersections[2])

        # print(len(a), len(b), len(inter1_data), len(inter2_data))
        # return abs(np.linalg.norm((a, inter1_data))) + \
        #        abs(np.linalg.norm((b, inter2_data))),
        return abs(sum((min(intersections[0]) - inter1_data_IRF)/inter1_data_IRF)) / len(inter1_data_IRF) + \
               abs(sum((max(intersections[0]) - inter2_data_IRF)/inter2_data_IRF)) / len(inter2_data_IRF) + \
               abs(sum((max(intersections[1]) - inter1_data_BCL)/inter1_data_BCL)) / len(inter1_data_BCL) + \
               abs(sum((min(intersections[1]) - inter2_data_BCL)/inter2_data_BCL)) / len(inter2_data_BCL) + \
               abs(sum((min(intersections[2]) - inter1_data_BLIMP)/inter1_data_BLIMP)) / len(inter1_data_BLIMP) + \
               abs(sum((max(intersections[2]) - inter2_data_BLIMP)/inter2_data_BLIMP)) / len(inter2_data_BLIMP)


    else:
        return 10000000


# @jit(nopython=True)
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
                 abs(np.array([np.random.normal(0.00005, 0.00025, pop_size), np.random.normal(0.1, 0.3, pop_size),
                               np.random.normal(2.6, 3, pop_size), np.random.normal(1, 1, pop_size),
                               np.random.normal(1, 1, pop_size), np.random.normal(2, 2, pop_size),
                               np.random.normal(100, 20, pop_size),
                               np.random.normal(1, 1, pop_size), np.random.normal(1, 1, pop_size),
                               np.random.normal(0.000001, 0.000001, pop_size),
                               np.random.normal(9, 5, pop_size),
                               np.random.normal(1, 1, pop_size), np.random.normal(1, 1, pop_size),# tot hier de
                               # waardes van de params, rest is sigmas
                               np.random.normal(0, 20, pop_size),
                               np.random.normal(0, 20, pop_size), np.random.normal(0, 20, pop_size),
                               np.random.normal(0, 20, pop_size), np.random.normal(0, 20, pop_size),
                               np.random.normal(0, 20, pop_size), np.random.normal(0, 20, pop_size),
                               np.random.normal(0, 20, pop_size), np.random.normal(0, 20, pop_size),
                               np.random.normal(0, 20, pop_size), np.random.normal(0, 20, pop_size),
                               np.random.normal(0, 20, pop_size), np.random.normal(0, 20, pop_size)]).T)
    return population


def initializer(d1_IRF, d2_IRF, d3_IRF, d1_BCL, d2_BCL, d3_BCL, d1_BLIMP, d2_BLIMP, d3_BLIMP):
    global inter1_data_IRF
    global inter2_data_IRF
    global inter3_data_IRF
    global inter1_data_BCL
    global inter2_data_BCL
    global inter3_data_BCL
    global inter1_data_BLIMP
    global inter2_data_BLIMP
    global inter3_data_BLIMP
    global tau
    global mutation_chance

    inter1_data_IRF = d1_IRF
    inter3_data_IRF = d3_IRF
    inter2_data_IRF = d2_IRF

    inter1_data_BCL = d1_BCL
    inter3_data_BCL = d3_BCL
    inter2_data_BCL = d2_BCL

    inter1_data_BLIMP = d1_BLIMP
    inter3_data_BLIMP = d3_BLIMP
    inter2_data_BLIMP = d2_BLIMP

    mutation_chance = 0.5
    tau = 0.90


def run_evolutionary_algo(pop_size, num_variables, num_gen, tournament_size,
                          inter1_data_IRF, inter2_data_IRF, inter3_data_IRF,
                          inter1_data_BCL, inter2_data_BCL, inter3_data_BCL,
                          inter1_data_BLIMP, inter2_data_BLIMP, inter3_data_BLIMP):
    """
    Run evolutionary algorithm in parallel
    """

    len_ind = num_variables * 2

    population = init_pop(pop_size=pop_size, num_variables=len_ind)
    # pool_input = [tuple(ind) for ind in population]

    best_sol_current = None
    best_fit_current = 100000000000
    mu_r_list = np.zeros(num_gen * pop_size)
    sigma_r_list = np.zeros(num_gen * pop_size)
    k_r_list = np.zeros(num_gen * pop_size)
    lambda_r_list = np.zeros(num_gen * pop_size)

    mu_b_list = np.zeros(num_gen * pop_size)
    sigma_b_list = np.zeros(num_gen * pop_size)
    k_b_list = np.zeros(num_gen * pop_size)
    lambda_b_list = np.zeros(num_gen * pop_size)


    mu_p_list = np.zeros(num_gen * pop_size)
    sigma_p_list = np.zeros(num_gen * pop_size)
    k_p_list = np.zeros(num_gen * pop_size)
    lambda_p_list = np.zeros(num_gen * pop_size)

    CD40_list = np.zeros(num_gen * pop_size)
    fitness_list = np.zeros(num_gen * pop_size)

    # start evolutionary algorithm
    for gen in range(num_gen):
        start_time = time.time()
        pool_input = tuple(population)

        # run the different solutions in parallel
        pool = Pool(cpu_count(), initializer, (inter1_data_IRF, inter2_data_IRF, inter3_data_IRF,
                          inter1_data_BCL, inter2_data_BCL, inter3_data_BCL,
                          inter1_data_BLIMP, inter2_data_BLIMP, inter3_data_BLIMP))
        fitnesses = pool.map(fitness, pool_input)
        pool.close()
        pool.join()


        # save all data about the population
        CD40_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 0]

        mu_r_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 1]
        sigma_r_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 2]
        k_r_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 3]
        lambda_r_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 4]

        mu_b_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 5]
        sigma_b_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 6]
        k_b_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 7]
        lambda_b_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 8]

        mu_p_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 9]
        sigma_p_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 10]
        k_p_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 11]
        lambda_p_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = np.array(population)[:, 12]

        fitness_list[gen * len(fitnesses): (gen + 1) * len(fitnesses)] = fitnesses

        best_fit_gen = min(fitnesses)

        if best_fit_gen < best_fit_current:
            best_sol_current = population[fitnesses.index(min(fitnesses))]
            best_fit_current = min(fitnesses)

        if gen % 5 == 0:
            best_solution = full_model(*best_sol_current[:num_variables])
            best_solution.calc_zeropoints()
            best_solution.plot('Ours', gen=gen)

        population = crossover(population, fitnesses, tournament_size, best_sol_current)

        pool_input = tuple(population)

        # run the different solutions in parallel
        pool = Pool(cpu_count(), initializer, (inter1_data_IRF, inter2_data_IRF, inter3_data_IRF,
                          inter1_data_BCL, inter2_data_BCL, inter3_data_BCL,
                          inter1_data_BLIMP, inter2_data_BLIMP, inter3_data_BLIMP))
        population = pool.map(mutation, pool_input)
        pool.close()
        pool.join()

        print("Best fit and average fit at gen {}: \t {}, \t {}, time to run it {}".format(gen, best_fit_gen,
                                                                                           np.mean(fitnesses),
                                                                                           time.time() - start_time))

    results = pd.DataFrame(data=np.array([CD40_list, mu_r_list, sigma_r_list, k_r_list, lambda_r_list,
                                          mu_b_list, sigma_b_list, k_b_list, lambda_b_list, mu_p_list,
                                          sigma_p_list, k_p_list, lambda_p_list, fitness_list]).T,
                           columns=['CD40', 'mu_r', 'sigma_r', 'k_r', 'lambda_r', 'mu_b', 'sigma_b', 'k_b', 'lambda_b',
                                    'mu_p', 'sigma_p', 'k_p', 'lambda_p', 'fitness'])
    results.to_csv('total_fitting_individuals_inclkandl.csv')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scat = ax.scatter3D(betas_list, p_list, zs=fitness_list, c=fitness_list,  cmap='hot', marker='.')
    # ax.set_xlabel('beta', fontweight='bold')
    # ax.set_ylabel('p', fontweight='bold')
    # ax.set_zlabel('fitness', fontweight='bold')
    # ax.set_xlim([0,12])
    # ax.set_ylim([0,0.1])
    # ax.set_zlim([0,12])
    # fig.colorbar(scat, ax=ax)
    #
    # plt.title("Fitnesses in Solution Space")
    # plt.show()

    return best_fit_current, best_sol_current


if __name__ == '__main__':
    # tau = 0.99   # needs to be changed in initializer function
    # mutation_chance = 0.25  # needs to be changed in initializer function
    crossover_chance = 0.25
    frac_to_replace = 0.5
    number_of_variables_to_fit = 13
    tournament_size = 100
    len_ind = number_of_variables_to_fit * 2  # times two for the sigmas
    pop_size = 10000
    num_gen = 30

    affymetrix_df = pd.read_csv('matrinez_data.csv')  # change this in the initializer as well
    # affymetrix_df = pd.read_csv('wesenhagen_data.csv')
    #
    affymetrix_df = affymetrix_df.set_index('Sample')
    #
    affymetrix_df = affymetrix_df.divide(4)

    inter1_data_IRF = np.append(affymetrix_df.loc['CB', 'IRF4'].values, affymetrix_df.loc['CC', 'IRF4'].values)
    inter2_data_IRF = affymetrix_df.loc['PC', 'IRF4'].values
    inter3_data_IRF = affymetrix_df.loc['CB', 'IRF4'].values

    inter1_data_BCL = np.append(affymetrix_df.loc['CB', 'BCL6'].values, affymetrix_df.loc['CC', 'BCL6'].values)
    inter2_data_BCL = affymetrix_df.loc['PC', 'BCL6'].values
    inter3_data_BCL = affymetrix_df.loc['CB', 'BCL6'].values

    inter1_data_BLIMP = np.append(affymetrix_df.loc['CB', 'PRDM1'].values, affymetrix_df.loc['CC', 'PRDM1'].values)
    inter2_data_BLIMP = affymetrix_df.loc['PC', 'PRDM1'].values
    inter3_data_BLIMP = affymetrix_df.loc['CB', 'PRDM1'].values


    best_fitness, best_ind = run_evolutionary_algo(pop_size, number_of_variables_to_fit, num_gen, tournament_size,
                                                   inter1_data_IRF, inter2_data_IRF, inter3_data_IRF,
                                                   inter1_data_BCL, inter2_data_BCL, inter3_data_BCL,
                                                   inter1_data_BLIMP, inter2_data_BLIMP, inter3_data_BLIMP)

    # def __init__(CD40, mu_r, sigma_r, mu_b, sigma_b, mu_p, sigma_p):

    best_solution = full_model(*best_ind[:number_of_variables_to_fit])
    # best_solution = IRF4(24.024979315717964, 0.05407750794340739, 204.30287870879206, 10.45117254171176,
    #                      12.432905080247275)

    print("De snijpunten met de x-as: ", best_solution.calc_zeropoints())
    print('mu_r: {}, sigma_r: {}, k_r: {}, l_r:{}, mu_b: {}, sigma_b: {}, k_b: {}, l_b:{}, mu_p: {}, sigma_p: {}, '
          'k_p: {}, l_p:{},'
          'CD40: {} '.format(
        best_solution.mu_r,
          best_solution.sigma_r, best_solution.k_r, best_solution.l_r,best_solution.mu_b, best_solution.sigma_b,
        best_solution.k_b, best_solution.l_b,
        best_solution.mu_p, best_solution.sigma_p, best_solution.k_p, best_solution.l_p,
          best_solution.CD40))
    print("The fitness of our solution: ", fitness([best_solution.CD40, best_solution.mu_r, best_solution.sigma_r,
                                                    best_solution.k_r, best_solution.l_r,
                                                    best_solution.mu_b, best_solution.sigma_b,
                                                    best_solution.k_b, best_solution.l_b,
                                                    best_solution.mu_p, best_solution.sigma_p,
                                                    best_solution.k_p, best_solution.l_p, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0]))

    best_solution.plot('Ours', gen=num_gen)

    mu_r = 0.1
    sigma_r = 2.6
    CD40 = 0.00001
    lambda_r = 1
    k_r = 1
    k_p = 1
    k_b = 1
    l_r = 1
    l_b =1
    l_p = 1
    beta = mu_r + CD40 + sigma_r / (lambda_r * k_r)
    p = -sigma_r / (lambda_r * k_r) + beta
    # beta = 12.76144062324778
    # p = 0.004582577456956276
    mu_b =2
    sigma_b = 100
    mu_p = 10**(-6)
    sigma_p = 9
    sol_of_martinez = full_model(CD40, mu_r, sigma_r, k_r, l_r, mu_b, sigma_b, k_b, l_b, mu_p, sigma_p, k_p, l_p)


    print('Martinez: mu: {}, sigma: {}, k: {}, lambda: {}, CD40: {} '.format(mu_r, CD40, sigma_r, lambda_r, k_r))
    print("Location of the roots: ", sol_of_martinez.calc_zeropoints())
    print("The fitness of the martinez solution: ", fitness([CD40, mu_r, sigma_r, k_r, l_r, mu_b, sigma_b, k_b,
                                                             l_b, mu_p,
                                                             sigma_p, k_p, l_p, 0,0, 0,
                                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    sol_of_martinez.plot('Martinez')
