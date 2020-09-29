import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random

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


class CD40SubNetwork(object):
    """ A class containing the parameters, equations and necessary functions for the standard Martinez model """

    def __init__(self, mu_p, sigma_p, mu_b, sigma_b, mu_r, sigma_r):

        self.mu_p = mu_p
        self.mu_b = mu_b
        self.mu_r = mu_r

        self.sigma_p = sigma_p
        self.sigma_b = sigma_b
        self.sigma_r = sigma_r

        self.k_p = 1
        self.k_b = 1
        self.k_r = 1

        self.lambda_p = 1
        self.lambda_b = 1
        self.lambda_r = 1

        self.b0 = 5
        self.p0 = 0
        self.r0 = 0

        self.cd0_signal = 50
        self.bcr0_signal = 53

        self.t_BCR_begin = 65
        self.t_BCR_end = 65.1

        self.t_cd40_begin = 70.1
        self.t_cd40_end = 70.2

    def equations(self, y, t, k):
        b, p, r = y

        bcr0 = 0
        cd0 = 0

        if self.t_cd40_end >= t >= self.t_cd40_begin:
            cd0 = self.cd0_signal

        BCR = bcr0 * self.k_b ** 2 / (self.k_b ** 2 + b ** 2)
        CD40 = cd0 * self.k_b ** 2 / (self.k_b ** 2 + b ** 2)

        self.BCR_list[k] = BCR
        self.CD40_list[k] = CD40

        dpdt = self.mu_p + self.sigma_p * self.k_b ** 2 / (self.k_b ** 2 + b ** 2) + self.sigma_p * \
               r ** 2 / (self.k_r ** 2 + r ** 2) - self.lambda_p * p

        dbdt = self.mu_b + self.sigma_b * self.k_b ** 2 / (self.k_b ** 2 + b ** 2) * self.k_p ** 2 / \
               (self.k_p ** 2 + p ** 2) * self.k_r ** 2 / (self.k_r ** 2 + r ** 2) - \
               (self.lambda_b + BCR) * b

        drdt = self.mu_r + self.sigma_r * r ** 2 / (self.k_r ** 2 + r ** 2) + CD40 - self.lambda_r * r

        return [dbdt, dpdt, drdt]

    def time_evolution(self, t_start, t_end, t_step):
        self.t_list = np.arange(t_start, t_end, t_step)

        self.BCR_list = np.zeros(len(self.t_list))
        self.CD40_list = np.zeros(len(self.t_list))

        solver_object = ForwardEuler(self.equations)
        solver_object.set_initial_condition([self.b0, self.p0, self.r0])
        self.soln, self.t_list = solver_object.solve(self.t_list)

        return self.soln, self.t_list

    def plot_evolution(self):
        b = self.soln[:, 0]
        p = self.soln[:, 1]
        r = self.soln[:, 2]

        plt.figure()
        plt.plot(self.t_list, p, label='BLIMP1', color='black')
        plt.plot(self.t_list, b, label='BCL-6', color='red')
        plt.plot(self.t_list, r, label='IRF-4', color='green')
        plt.plot(self.t_list, self.BCR_list, label='BCR', color='orange')
        plt.plot(self.t_list, self.CD40_list, label='CD40', color='blue')
        plt.ylabel('Concentration')
        plt.xlabel('Time')
        plt.legend()
        plt.grid()
        plt.title('Concentration over time')
        plt.show()


def fitness(ind):
    '''
    Calculates the fitness of an individual as the mse on the time series
    :param ind: an array containing the parameters we want to be fitting
    :return: the fitness of an individual, the lower the better tho
    '''

    # instantiate an individual
    model_ind = CD40SubNetwork(*ind)

    # calculate the concentration development over time for the given parameters
    t_start = 0
    t_end = 100
    t_step = 0.01
    soln, t_list = model_ind.time_evolution(t_start, t_end, t_step)

    # we have very little data points of the affymetrix data, but it does need to line up, so this is a first attempt
    # can probably be greatly improved tho
    t_list = np.linspace(t_start, t_end-1, len(affymetrix_df['BCL6'])) / t_step
    t_list_indices_affymetrix_data = list(map(int, t_list))

    # calculate the mse for the three components. If it one of the components has exploded into infinity, fitness is low
    mse = 0
    if sum(sum(np.isinf(soln))) == 0 and sum(sum(np.isnan(soln))) == 0:
        for (i, protein) in enumerate(['BCL6', 'PRDM1', 'IRF4']):
            mse += mean_squared_error(soln[t_list_indices_affymetrix_data, i], affymetrix_df[protein])
    else:
        mse = 100000

    return mse,


def generateES(ind_cls, strg_cls, size):

    ind = ind_cls(random.gauss(0, 10) for _ in range(size))
    ind.strategy = strg_cls(random.gauss(0,10) for _ in range(size))
    return ind


# this needs to be before the if __name__ blabla to make the multiprocessing work
affymetrix_df = pd.read_csv('affymetrix_data.csv')

creator.create("Strategy", np.ndarray)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
IND_SIZE = 6
toolbox.register("evaluate", fitness)
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=5)

if __name__ == '__main__':

    # plot figure showing the affymetrix data as if it is a time series
    ax = plt.gca()

    affymetrix_df.plot(kind='line', y='PRDM1', ax=ax, color='black')
    affymetrix_df.plot(kind='line', y='BCL6', ax=ax, color='red')
    affymetrix_df.plot(kind='line', y='IRF4', ax=ax, color='green')

    plt.text(2.01, 8, r'CB', color='black')
    plt.axvline(x=4, linestyle='--', color='black')
    plt.axvline(x=5, linestyle='--', color='purple')
    plt.text(6.01, 8, r'CC', color='purple')
    plt.axvline(x=9, linestyle='--', color='purple')
    plt.axvline(x=10, linestyle='--', color='red')
    plt.text(12.01, 8, r'PC', color='red')
    plt.show()

    # EA stuff starts

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    hof.clear()

    pop = toolbox.population(n=30)

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=19, lambda_=50, cxpb=0.6, mutpb=0.4,
                                              ngen=50, stats=stats, halloffame=hof, verbose=True)

    # show what the best solution looks like
    best_sol = CD40SubNetwork(*hof[0])
    t_start = 0
    t_end = 100
    t_step = 0.01
    soln, t_list = best_sol.time_evolution(t_start, t_end, t_step)
    best_sol.plot_evolution()
