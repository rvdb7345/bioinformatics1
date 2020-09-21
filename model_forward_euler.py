import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt


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
                 lambda_b, b0, p0, r0, cd0_signal, bcr0_signal, t_BCR_begin,
                 t_BCR_end, t_cd40_begin, t_cd40_end):

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

        self.t_BCR_begin = t_BCR_begin
        self.t_BCR_end = t_BCR_end

        self.t_cd40_begin = t_cd40_begin
        self.t_cd40_end = t_cd40_end

    def calc_beta(self):
        self.beta = (self.mu_r + self.cd0_signal + self.sigma_r) / (self.lambda_r * self.k_r)

    def equations(self, y, t, k):
        b, p, r = y

        bcr0 = 0
        cd0 = 0
        if self.t_BCR_end >= t >= self.t_BCR_begin:
            bcr0 = self.bcr0_signal
        elif self.t_cd40_end >= t >= self.t_cd40_begin:
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
        solver_object.set_initial_condition([b0, p0, r0])
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





if __name__ == '__main__':
    ''' 
        What do the parameters refer to:
         - mu: basal transcription rate
         - sigma: maximum induced transcription rate
         - k: dissociation constant
         - lambda: degradation rate
    '''

    # parameters corresponding to BLIMP1 (p)
    mu_p = 1 ** (-6)
    sigma_p = 9.0
    k_p = 1.0
    lambda_p = 1.0

    # parameters corresponding to BCL-6 (b)
    mu_b = 2
    sigma_b = 100
    k_b = 1.0
    lambda_b = 1.0

    # parameters corersponding to IRF-4 (r)
    mu_r = 0.1
    sigma_r = 2.6
    k_r = 1.0
    lambda_r = 1.0

    # parameters characterising the initial state
    b0 = 5
    p0 = 0
    r0 = 0
    cd0_signal = 20
    bcr0_signal = 5

    t_start = -20
    t_end = 50
    t_step = 0.001

    t_BCR_begin = 4
    t_BCR_end = 4.1

    t_cd40_begin = 5
    t_cd40_end = 5.125


    # instantiate the model with the correct parameters
    model = MartinezModel(mu_p, sigma_p, k_p, lambda_p, mu_b, sigma_b, k_b,
                          lambda_b, b0, p0, r0, cd0_signal, bcr0_signal, t_BCR_begin,
                          t_BCR_end, t_cd40_begin, t_cd40_end)

    soln, t_list = model.time_evolution(t_start, t_end, t_step)
    model.plot_evolution()
    model.beta_phase_plot()
