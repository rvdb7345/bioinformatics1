import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from numba import jit
import tqdm
from scipy.stats import norm

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


class MartinezModel(object):
    """ A class containing the parameters, equations and necessary functions for the standard Martinez model """


    def __init__(self, mu_p, sigma_p, k_p, lambda_p, mu_b, sigma_b, k_b,
                 lambda_b, mu_r, sigma_r, k_r, lambda_r, b0, p0, r0, cd0_signal, 
                 bcr0_signal, t_BCR_mean, t_BCR_std, t_cd40_mean, t_cd40_std):

        self.mu_p = mu_p
        self.mu_b = mu_b
        self.mu_r = mu_r

        self.sigma_p = sigma_p
        self.sigma_b = sigma_b
        self.sigma_r = sigma_r

        self.k_p = k_p
        self.k_b = k_b
        self.k_r = k_r

        self.lambda_p = lambda_p
        self.lambda_b = lambda_b
        self.lambda_r = lambda_r

        self.b0 = b0
        self.p0 = p0
        self.r0 = r0

        self.cd0_signal = cd0_signal
        self.bcr0_signal = bcr0_signal

        self.t_BCR_mean = t_BCR_mean
        self.t_BCR_std = t_BCR_std

        self.t_cd40_mean = t_cd40_mean
        self.t_cd40_std = t_cd40_std

    def calc_beta(self):
        self.beta = (self.mu_r + self.cd0_signal + self.sigma_r) / (self.lambda_r * self.k_r)

    def equations(self, y, t, k):
        b, p, r = y

        # bcr0 = 0
        # cd0 = 0
        # elif self.t_cd40_end >= t >= self.t_cd40_begin:
        #     cd0 = self.cd0_signal

        bcr0 = self.bcr0_signal * norm.pdf(t, self.t_BCR_mean, self.t_BCR_std)
        cd0 = self.cd0_signal * norm.pdf(t, self.t_cd40_mean, self.t_cd40_std)

        if t == self.t_BCR_mean:
            print("bcr0 at peak = ", bcr0)
        if t == self.t_cd40_mean:
            print("cd0  at peak = ", cd0)

        BCR = bcr0 * self.k_b ** 2 / (self.k_b ** 2 + b ** 2)
        CD40 = cd0 * self.k_b ** 2 / (self.k_b ** 2 + b ** 2)

        self.BCR_list[k] = BCR
        self.CD40_list[k] = CD40

        dpdt = self.mu_p + self.sigma_p * self.k_b ** 2 / (self.k_b ** 2 + b ** 2) + \
               self.sigma_p * r ** 2 / (self.k_r ** 2 + r ** 2) - self.lambda_p * p

        dbdt = self.mu_b + self.sigma_b * \
               self.k_b ** 2 / (self.k_b ** 2 + b ** 2) * \
               self.k_p ** 2 / (self.k_p ** 2 + p ** 2) * \
               self.k_r ** 2 / (self.k_r ** 2 + r ** 2) - \
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

        fig = plt.figure()
        plt.plot(self.t_list, p, label='BLIMP1', color='black')
        plt.plot(self.t_list, b, label='BCL-6', color='red')
        plt.plot(self.t_list, r, label='IRF-4', color='green')
        plt.plot(self.t_list, self.BCR_list, 'y--', label='BCR')
        plt.plot(self.t_list, self.CD40_list, 'b--', label='CD40')
        plt.ylabel('Concentration ($10^{-8}$ M)')
        plt.xlabel('Time (4h)')
        plt.legend()
        plt.grid()
        plt.title('Concentration over time')
        fig.savefig("TimeEvolutionModel.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10,3))
        plt.plot(self.t_list, self.BCR_list, label='BCR', color='orange')
        plt.plot(self.t_list, self.CD40_list, label='CD40', color='blue')
        plt.ylabel('Concentration ($10^{-8}$ M)')
        plt.xlabel('Time (4h)')
        plt.xlim(self.t_BCR_mean - 3 * self.t_BCR_std, self.t_cd40_mean + 3 * self.t_cd40_std)
        plt.legend()
        plt.grid()
        plt.title('Concentration over time')
        fig.savefig("BCR_CD40.png")
        plt.close(fig)

    def beta_phase_plot(self):
        plt.figure()
        plt.plot(self.BCR_list)
        plt.show()

        plt.figure()
        plt.plot(self.BCR_list, self.soln[:, 0])
        plt.grid()
        plt.xlabel('BCR')
        plt.ylabel('BCL6')
        plt.show()

    def sensitivity_analysis(self, t_start, t_end, t_step, y_param, x_parameter, x_values):
        if y_param in ['b', 'bcl6', 'BCL6']:
            y_index = 0
        elif y_param in ['p', 'blimp1', 'BLIMP1']:
            y_index = 1
        elif y_param in ['r', 'irf4', 'IRF4']:
            y_index = 2
        else:
            print('y parameter not recognised')

        end_values = np.zeros(len(x_values))

        for i, param_value in enumerate(x_values):
            setattr(self, x_parameter, param_value)
            self.time_evolution(t_start, t_end, t_step)
            # self.plot_evolution()
            end_values[i] = self.soln[-1, y_index]

        plt.figure()
        plt.scatter(x_values, end_values)
        plt.xlabel(x_parameter)
        plt.ylabel(y_param)
        plt.grid()
        plt.show()


class BCRSubNetwork(MartinezModel):
    def equations(self, y, t, k):
        if self.nullcline_plot_bcl6:
            b, p, r = y

            BCR = self.BCR
            CD40 = 0

            b = self.b0

            dpdt = self.mu_p + self.sigma_p * self.k_b ** 2 / (self.k_b ** 2 + b ** 2) - self.lambda_p * p

            dbdt = 0

            drdt = self.mu_r + self.sigma_r * r ** 2 / (self.k_r ** 2 + r ** 2) + CD40 - self.lambda_r * r

        if self.nullcline_plot_blimp1:
            b, p, r = y

            BCR = self.BCR
            CD40 = 0

            p = self.p0

            dpdt = 0

            dbdt = self.mu_b + self.sigma_b * self.k_b ** 2 / (self.k_b ** 2 + b ** 2) * self.k_p ** 2 / \
               (self.k_p ** 2 + p ** 2) - (self.lambda_b + BCR) * b

            drdt = self.mu_r + self.sigma_r * r ** 2 / (self.k_r ** 2 + r ** 2) + CD40 - self.lambda_r * r

        return [dbdt, dpdt, drdt]

    def nullclines_trajectories(self, BCR, t_start, t_end, t_step):
        self.nullcline_plot_blimp1 = False
        self.nullcline_plot_bcl6 = True
        self.BCR = BCR

        BCL6_list = np.arange(0, 5.5, 0.25)

        blimp1_end_values = np.zeros(len(BCL6_list))

        for (i, bcl6) in enumerate(BCL6_list):
            self.b0 = bcl6
            self.time_evolution(t_start, t_end, t_step)
            blimp1_end_values[i] = self.soln[-1, 1]

        self.nullcline_plot_bcl6 = False
        self.nullcline_plot_blimp1 = True

        BLIMP1_list = np.arange(0, 5.5, 0.25)

        bcl6_end_values = np.zeros(len(BLIMP1_list))

        for (i, blimp1) in enumerate(BLIMP1_list):
            self.p0 = blimp1
            self.time_evolution(t_start, t_end, t_step)
            # self.plot_evolution()
            bcl6_end_values[i] = self.soln[-1, 0]

        plt.figure()
        plt.scatter(BCL6_list, blimp1_end_values, label='BCL6 fixed')
        plt.scatter(bcl6_end_values, BLIMP1_list, label='BLIMP1 fixed')
        plt.xlabel('BCL6')
        plt.ylabel('BLIMP1')
        plt.legend()
        plt.show()



class CD40SubNetwork(MartinezModel):
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


if __name__ == '__main__':
    ''' 
        What do the parameters refer to:
         - mu: basal transcription rate
         - sigma: maximum induced transcription rate
         - k: dissociation constant
         - lambda: degradation rate
    '''
    

    """
    MARTINEZ PARAMS:
    Paramaters according to Martinez (2015), table S2.
    """
    # # parameters corresponding to BLIMP1 (p)
    # mu_p = 10 ** (-6)
    # sigma_p = 9.0
    # k_p = 1.0
    # lambda_p = 1.0
    #
    # # parameters corresponding to BCL-6 (b)
    # mu_b = 2
    # sigma_b = 100
    # k_b = 1.0
    # lambda_b = 1.0
    #
    # # parameters corersponding to IRF-4 (r)
    # mu_r = 0.1
    # sigma_r = 2.6
    # k_r = 1.0
    # lambda_r = 1.0
    #
    # # parameters characterising the initial state
    # b0 = 5
    # p0 = 0
    # r0 = 0
    # cd0_signal = 10
    # bcr0_signal = 13

    """
    FOUND PARAMS LIGHT:
    Parameters mu and sigma fitted, with data / 4
    """
    # # Basal transcription rate
    # mu_p = 1.509e-6
    # mu_b = 1.514
    # mu_r = 0.09513
    # # Maximum induced transcription rate
    # sigma_p = 2.188
    # sigma_b = 65.64
    # sigma_r = 2.727
    # # Dissociation constant
    # k_p = 1
    # k_b = 1
    # k_r = 1
    # # Degradation rate
    # lambda_p = 1
    # lambda_b = 1
    # lambda_r = 1

    """
    FOUND PARAMS EXTREME:
    All parameters fitted (also lambda en k)
    """
    # Basal transcription rate
    mu_p = 1.107e-6
    mu_b = 3.990
    mu_r = 0.2279
    # Maximum induced transcription rate
    sigma_p = 6.275
    sigma_b = 118.5
    sigma_r = 4.776
    # Dissociation constant
    k_p = 3.940
    k_b = 4.199
    k_r = 5.290
    # Degradation rate
    lambda_p = 0.674
    lambda_b = 1.731
    lambda_r = 0.4011


    # parameters characterising the initial state
    b0 = 10
    p0 = 0
    r0 = 0

    t_start = 0
    t_end = 200
    t_step = 0.01

    # Gaussian bcr0 and cd0 signal settings
    bcr0_signal = 45 # check (in output) if bcr0 at peak <10
    cd0_signal = 8 # check (in output) if cd0 at peak <1

    t_BCR_mean = 40
    t_BCR_std = 7

    t_cd40_mean = 60
    t_cd40_std = 5


    MartinezModel = MartinezModel(mu_p, sigma_p, k_p, lambda_p, mu_b, sigma_b, k_b,
                          lambda_b, mu_r, sigma_r, k_r, lambda_r, b0, p0, r0, cd0_signal, 
                          bcr0_signal, t_BCR_mean, t_BCR_std, t_cd40_mean, t_cd40_std)

    MartinezModel.time_evolution(t_start, t_end, t_step)
    MartinezModel.plot_evolution()