''' Authors: Katinka den Nijs & Robin van den Berg '''

import multiprocessing
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import cma
from scipy.optimize import fsolve, root
import random
import math
import time
from numba import jit
from tqdm import tqdm


class params_article():
    """ Standard param settings (according to Martinez (2015)) """

    def __init__(self):
        # Basal transcription rate
        self.mu_p = 10e-6
        self.mu_b = 2
        self.mu_r = 0.1
        # Maximum induced transcription rate
        self.sigma_p = 9
        self.sigma_b = 100
        self.sigma_r = 2.6
        # Dissociation constant
        self.k_p = 1
        self.k_b = 1
        self.k_r = 1
        # Degradation rate
        self.l_p = 1
        self.l_b = 1
        self.l_r = 1


class BLIMP1(params_article):
    """ Class containing BlIMP1 specific equations according to Martinez (2015) """

    def __init__(self, mu_p, sigma_p):
        super().__init__()
        self.mu_p = mu_p
        self.sigma_p = sigma_p

        self.b = GC_data_BCL6
        self.r = GC_data_IRF4

    def change_stage(self):
        # to switch between b/r-values of PC and GC stage
        if np.all(self.b == GC_data_BCL6):
            self.b = PC_data_BCL6
            self.r = PC_data_IRF4
        elif np.all(self.b == PC_data_BCL6):
            self.b = GC_data_BCL6
            self.r = GC_data_IRF4

    def equation(self, p):
        dpdt = self.mu_p + (self.sigma_p * self.k_b ** 2) / (self.k_b ** 2 + np.mean(self.b) ** 2) + (
                    self.sigma_p * (np.mean(self.r)) ** 2) / (self.k_r ** 2 + np.mean(self.r) ** 2) - self.l_p * p
        return dpdt

    def calc_zeropoints(self, guesses=[0]):
        self.intersections = root(self.equation, guesses)
        return np.sort(self.intersections.x)

    def plot(self, dataset='test', fitting="Fitted Params"):
        p_list = np.arange(0, 10, 0.001)

        intersections_GC = self.calc_zeropoints(np.mean(GC_data_BLIMP))
        dpdt_GC = self.equation(p_list)

        self.change_stage()

        intersections_PC = self.calc_zeropoints(np.mean(PC_data_BLIMP))
        dpdt_PC = self.equation(p_list)

        ### STILL INCLUDE UNITS
        fig = plt.figure()
        plt.title("BLIMP rate equation fit {}-data with {}\nIntersections: {:.2f}, {:.2f}".format(dataset, fitting,
                                                                                                  intersections_GC[0],
                                                                                                  intersections_PC[0]))
        plt.plot(p_list, dpdt_GC, label="fit GC")
        plt.plot(p_list, dpdt_PC, label="fit PC")
        plt.scatter(GC_data_BLIMP, np.zeros(len(GC_data_BLIMP)), label="data GC")
        plt.scatter(PC_data_BLIMP, np.zeros(len(PC_data_BLIMP)), label="data PC")
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.legend(fontsize=12)
        plt.xlabel("BLIMP1", fontsize=15)
        plt.ylabel("dP/dt", fontsize=15)
        # plt.ylim(min(drdt), 3)
        # plt.xlim(0,8)
        fig.savefig("BLIMPDataFit{}{}.png".format(dataset.capitalize(), fitting.replace(" ", "")))
        plt.close(fig)


def fitness(ind):
    '''
    Calculates the fitness of an individual as the mse on the time series
    :param ind: an array containing the parameters we want to be fitting
    :return: the fitness of an individual, the lower the better tho
    '''

    model = BLIMP1(*ind)
    intersections_GC = model.calc_zeropoints(np.mean(GC_data_BLIMP))
    model.change_stage()
    intersections_PC = model.calc_zeropoints(np.mean(PC_data_BLIMP))

    return abs(sum(intersections_GC - GC_data_BLIMP)) + abs(sum(intersections_PC - PC_data_BLIMP)),


def mutate(ind, mu, sigma, indpb, c=1):
    """ Adapted mutate function of deap package; only non-negative values permitted """

    size = len(ind)
    t = c / math.sqrt(2. * math.sqrt(size))
    t0 = c / math.sqrt(2. * size)
    n = random.gauss(0, 1)
    t0_n = t0 * n

    for indx in range(size):
        if random.random() < indpb:
            ind.strategy[indx] *= math.exp(t0_n + t * random.gauss(0, 1))
            ind[indx] += ind.strategy[indx] * random.gauss(0, 1)
            ind[indx] = abs(ind[indx])

    return ind,


def generateES(ind_cls, strg_cls, size):
    ind = ind_cls(abs(random.gauss(0, 2)) for _ in range(size))
    ind.strategy = strg_cls(abs(random.gauss(0, 3)) for _ in range(size))
    return ind


def cxESBlend(ind1, ind2, alpha):
    """ Adapted cxESBlend of deap package; only non-negative values permitted """

    for i, (x1, s1, x2, s2) in enumerate(zip(ind1, ind1.strategy,
                                             ind2, ind2.strategy)):
        # Blend the values
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = abs((1. - gamma) * x1 + gamma * x2)
        ind2[i] = abs(gamma * x1 + (1. - gamma) * x2)
        # Blend the strategies
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1.strategy[i] = (1. - gamma) * s1 + gamma * s2
        ind2.strategy[i] = gamma * s1 + (1. - gamma) * s2

    return ind1, ind2


# This needs to be before the if __name__ statement to make the multiprocessing work

# Choose dataset
# dataset = "Martinez"
dataset = "Wesenhagen"
if dataset == "Martinez":
    affymetrix_df = pd.read_csv('martinez_data.csv')
    affymetrix_df = affymetrix_df.set_index('Sample')
else:
    affymetrix_df = pd.read_csv('wesenhagen_data.csv')
    affymetrix_df = affymetrix_df.set_index('Sample') / 4

# Create two groups per gene: GC and PC, before and after switch
GC_data_IRF4 = np.append(affymetrix_df.loc['CB', 'IRF4'].values, affymetrix_df.loc['CC', 'IRF4'].values)
PC_data_IRF4 = affymetrix_df.loc['PC', 'IRF4'].values
GC_data_BCL6 = np.append(affymetrix_df.loc['CB', 'BCL6'].values, affymetrix_df.loc['CC', 'BCL6'].values)
PC_data_BCL6 = affymetrix_df.loc['PC', 'BCL6'].values
GC_data_BLIMP = np.append(affymetrix_df.loc['CB', 'PRDM1'].values, affymetrix_df.loc['CC', 'PRDM1'].values)
PC_data_BLIMP = affymetrix_df.loc['PC', 'PRDM1'].values

# Initialization deap toolbox
creator.create("Strategy", np.ndarray)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
IND_SIZE = 2
toolbox.register("evaluate", fitness)
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                 IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", cxESBlend, alpha=0.1)
toolbox.register("mutate", mutate, mu=0, sigma=1, indpb=1)
toolbox.register("select", tools.selTournament, tournsize=7)

if __name__ == '__main__':
    print("BLIMP fitting")
    print("{}-data".format(dataset))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    pop = toolbox.population(n=10000)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=10000, lambda_=5000, cxpb=0.1, mutpb=0.90,
                                             ngen=30, stats=stats, halloffame=hof, verbose=True)

    best_sol = BLIMP1(*hof[0])

    best_sol.plot(dataset=dataset, fitting="Fitted Params")

    print('mu_p and sigma_p of our solution: {}, {}'.format(best_sol.mu_p, best_sol.sigma_p))
    print('Fitness of our best solution: ', fitness(hof[0])[0])

    params_mart = [10e-6, 9]

    sol_mart = BLIMP1(*params_mart)

    sol_mart.plot(dataset=dataset, fitting="Martinez Params")

    print('mu_p and sigma_p of Martinez solution: {}, {}'.format(sol_mart.mu_p, sol_mart.sigma_p))
    print('Fitness of Martinez solution: ', fitness(params_mart)[0])
