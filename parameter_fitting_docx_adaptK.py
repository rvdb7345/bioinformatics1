import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scipy.optimize import fsolve, root
import random
import math

class ForwardEuler(object):
    """
    Class for solving a scalar of vector ODE,

      du/dt = f(u, t)

    by the ForwardEuler solver.

    Class attributes:
    t: array of time values
    u: array of solution values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(u, t)
    """

    def __init__(self, f):
        if not callable(f):
            raise TypeError('f is %s, not a function' % type(f))
        self.f = lambda u, t, k: np.asarray(f(u, t, k))

    def set_initial_condition(self, U0):
        if isinstance(U0, (float, int)):  # scalar ODE
            self.neq = 1
        else:  # system of ODEs
            U0 = np.asarray(U0)
            self.neq = U0.size
        self.U0 = U0

    def solve(self, time_points):
        """Compute u for t values in time_points list."""
        self.t = np.asarray(time_points)
        n = self.t.size
        if self.neq == 1:  # scalar ODEs
            self.u = np.zeros(n)
        else:  # systems of ODEs
            self.u = np.zeros((n, self.neq))

        # Assume self.t[0] corresponds to self.U0
        self.u[0] = self.U0

        # Time loop
        for k in range(n - 1):
            self.k = k
            self.u[k + 1] = self.advance()
        return self.u, self.t

    def advance(self):
        """Advance the solution one time step."""
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k + 1] - t[k]
        u_new = u[k] + dt * f(u[k], t[k], k)
        u_new = [(i > 0) * i for i in u_new]

        return u_new


class IRF4(object):
    """ A class containing the parameters, equations and necessary functions for the standard Martinez model """

    def __init__(self, beta, p):
        self.beta = beta
        self.p = p

    def equation(self, y):
        # drdt = self.mu_r + self.sigma_r * r ** 2 / (self.k_r ** 2 + r ** 2) + CD40 - self.lambda_r * r

        F = y**3 - self.beta * y**2 + y - self.p

        return F

    def calc_zeropoints(self):
        self.intersections = root(self.equation, [0., 0.1, 10], method='lm')
        return np.sort(self.intersections.x)

    def plot(self, name='test'):
        intersections = self.intersections.x
        # intersections = np.sort(intersections)
        r_list = np.arange(0, 5, 0.001)
        drdt = self.equation(r_list)

        for i in drdt:
            print(i)

        # intersections = self.intersections.x
        fig = plt.figure()
        plt.title("{}: With intersections: {}".format(name, intersections))
        # plt.plot(r_list + intersections[1], drdt - drdt[0])
        plt.plot(r_list, drdt)
        plt.scatter(intersections, [0,0,0])
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.ylim(-3, 3)
        plt.xlim(0,8)
        fig.savefig("AffymetrixData{}.png".format(name.capitalize()))
        plt.close(fig)

def fitness(ind):
    '''
    Calculates the fitness of an individual as the mse on the time series
    :param ind: an array containing the parameters we want to be fitting
    :return: the fitness of an individual, the lower the better tho
    '''

    model_ind = IRF4(*ind)
    intersections = model_ind.calc_zeropoints()
    delta_intersec = abs(intersections[2] - intersections[0])

    # instantiate an individual
    if (ind[0] ** 2 > 3) and (ind[0] ** 3 + (ind[0] ** 2 - 3) ** (3/2) - 9 * ind[0] / 2 > - 27/2 * ind[1]) and \
            (ind[0] ** 3 - (ind[0] ** 2 - 3) ** (3 / 2) - 9 * ind[0] / 2 < - 27/2 * ind[1]) and \
            (delta_intersec > 0.1):

        return abs(sum(intersections[0] - inter1_data)) + abs(sum(intersections[2] - inter2_data)),

    else:
        return 10000000,

def generateES(ind_cls, strg_cls, size):
    ind = ind_cls(abs(random.gauss(0, 2)) for _ in range(size))
    ind.strategy = strg_cls(abs(random.gauss(0, 3)) for _ in range(size))
    return ind

def mutate(ind, mu, sigma, indpb, c=1):
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


# this needs to be before the if __name__ blabla to make the multiprocessing work
affymetrix_df = pd.read_csv('matrinez_data.csv')
# affymetrix_df = pd.read_csv('wesenhagen_data.csv')

# # replaces df by average of all rows per sample
# affymetrix_df = affymetrix_df.groupby('Sample').agg({'PRDM1':'mean','BCL6':'mean','IRF4':'mean'})
affymetrix_df=affymetrix_df.set_index('Sample')
inter1_data = np.append(affymetrix_df.loc['CB', 'IRF4'].values, affymetrix_df.loc['CC', 'IRF4'].values)
inter2_data = affymetrix_df.loc['PC', 'IRF4'].values

creator.create("Strategy", np.ndarray)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
IND_SIZE = 2
toolbox.register("evaluate", fitness)
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", mutate, mu=0, sigma=1, indpb=1)
toolbox.register("select", tools.selTournament, tournsize=7)

if __name__ == '__main__':
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    # hof.clear()

    pop = toolbox.population(n=10000)

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=10000, lambda_=5000, cxpb=0.1, mutpb=0.90,
                                              ngen=50, stats=stats, halloffame=hof, verbose=True)


    # show what the best solution looks like
    best_sol = IRF4(*hof[0])
    print('Beta and p of our solution: {}, {}'.format(best_sol.beta, best_sol.p))
    sigma_r = best_sol.beta - best_sol.p
    mu_r = best_sol.beta - sigma_r
    print('sigma_r and mu_r of our solution: {}, {}'.format(sigma_r, mu_r))
    print("Location of the roots: ", best_sol.calc_zeropoints())
    print('Fitness of our best solution: ', fitness(hof[0])[0])

    best_sol.plot('ours')

    mu_r = 0.1
    sigma_r = 2.6
    CD40 = 0
    lambda_r = 1
    k_r = 1
    beta = mu_r + CD40 + sigma_r / (lambda_r * k_r)
    p = -sigma_r/(lambda_r*k_r) + beta
    sol_of_martinez = IRF4(beta, p)

    print('Beta and p of martinez: {}, {}'.format(beta, p))
    print('sigma_r and mu_r of martinez: {}, {}'.format(sigma_r, mu_r))
    print("Location of the roots: ", sol_of_martinez.calc_zeropoints())
    print("The fitness of the martinez solution: ", fitness([beta, p])[0])

    sol_of_martinez.plot('martinez')
