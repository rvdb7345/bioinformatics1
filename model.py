import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

global iterator
iterator=0
bcr_list = []
cd0_list = []
t_list_inside = []

def equations(t, y, bcr0, cd0, mu_r, mu_b, mu_p, sigma_p, sigma_b, sigma_r, k_r, k_b, k_p, lambda_r, lambda_b, lambda_p):
    b, p, r = y

    global iterator
    global bcr_list
    global cd0_list
    global t_list_inside

    if iterator == 50:
        bcr0 = 50000
        cd0 = 100
    else:
        bcr0=0
        cd0=0

    BCR = bcr0 * k_b ** 2 / (k_b ** 2 + b ** 2)
    CD40 = cd0 * k_b ** 2 / (k_b ** 2 + b ** 2)

    dpdt = mu_p + sigma_p * k_b ** 2 / (k_b ** 2 + b ** 2) + sigma_p * \
           r ** 2 / (k_r ** 2 + r ** 2) - lambda_p * p

    dbdt = mu_b + sigma_b * k_b ** 2 / (k_b ** 2 + b ** 2) * k_p ** 2 / \
           (k_p ** 2 + p ** 2) * k_r ** 2 / (k_r ** 2 + r ** 2) - \
           (lambda_b + BCR) * b

    drdt = mu_r + sigma_r * r ** 2 / (k_r ** 2 + r ** 2) + CD40 - lambda_r * r

    print(dpdt, dbdt, drdt)


    iterator += 1

    bcr_list.append(bcr0)
    cd0_list.append(cd0)
    t_list_inside.append(t)

    return [dpdt, dbdt, drdt]

def equations_odeint(y, t, bcr0, cd0, mu_r, mu_b, mu_p, sigma_p, sigma_b, sigma_r, k_r, k_b, k_p, lambda_r, lambda_b, lambda_p):
    b, p, r = y

    BCR = bcr0 * k_b ** 2 / (k_b ** 2 + b ** 2)
    CD40 = cd0 * k_b ** 2 / (k_b ** 2 + b ** 2)

    dpdt = mu_p + sigma_p * k_b ** 2 / (k_b ** 2 + b ** 2) + sigma_p * \
           r ** 2 / (k_r ** 2 + r ** 2) - lambda_p * p

    dbdt = mu_b + sigma_b * k_b ** 2 / (k_b ** 2 + b ** 2) * k_p ** 2 / \
           (k_p ** 2 + p ** 2) * k_r ** 2 / (k_r ** 2 + r ** 2) - \
           (lambda_b + BCR) * b

    drdt = mu_r + sigma_r * r ** 2 / (k_r ** 2 + r ** 2) + CD40 - lambda_r * r

    return [dpdt, dbdt, drdt]


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
    bcr0 = 4
    cd0 = 5
    b0 = 0
    p0 = 5
    r0 = 0

    t_start = 0
    t_end = 5
    t_step = 0.01
    t_steps = 0.1

    t_list = np.arange(t_start, t_end, t_step)
    # bcr_list = np.zeros(t_steps)
    # bcr_list[1] = 1
    # cd0_list = np.zeros(t_steps)
    # cd0_list[1] = 5

    # soln = odeint(equations_odeint, [b0, p0, r0], t_list,  args=(bcr0, cd0, mu_r, mu_b, mu_p, sigma_p, sigma_b, sigma_r, k_r, k_b, k_p,
    #                                         lambda_r, lambda_b, lambda_p))

    soln = solve_ivp(fun=equations, t_span=[t_start, t_end], y0=[b0, p0, r0],
                     method='LSODA', args=(bcr0, cd0, mu_r, mu_b, mu_p, sigma_p, sigma_b, sigma_r, k_r, k_b, k_p,
                                           lambda_r, lambda_b, lambda_p), t_eval=t_list, rtol = 1e-5)

    # soln = {'y': np.array([[p0], [b0], [r0]])}
    #
    # total_sol = np.array([[p0], [b0], [r0]])
    #
    # t = [t_start]
    #
    # for i in range(0, t_steps-1):
    #     p0 = soln['y'][0, -1]
    #     b0 = soln['y'][1, -1]
    #     r0 = soln['y'][2, -1]
    #     brc0 = bcr_list[i]
    #     cd0 = cd0_list[i]
    #
    #     soln = solve_ivp(fun=equations, t_span=[t_list[i], t_list[i+1]], y0=[b0, p0, r0], dense_output=True,
    #                      method='LSODA', args=(bcr0, cd0, mu_r, mu_b, mu_p, sigma_p, sigma_b, sigma_r, k_r, k_b, k_p,
    #                                            lambda_r, lambda_b, lambda_p))
    #
    #     total_sol = np.concatenate((total_sol, soln['y']), axis=1)
    #     t = np.concatenate((t, soln['t']))
    #
    #
    #
    # soln = total_sol
    # print(soln)
    #

    # p = soln[:, 0]
    # b = soln[:, 1]
    # r = soln[:, 0]

    # p = soln[0, :]
    # b = soln[1, :]
    # r = soln[2, :]
    p = soln['y'][0, :]
    b = soln['y'][1, :]
    r = soln['y'][2, :]

    plt.figure()
    # plt.plot(soln['t'], p, label='BLIMP1', color='black')
    # plt.plot(soln['t'], b, label='BCL-6', color='red')
    # plt.plot(soln['t'], r, label='IRF-4', color='green')
    plt.plot(t_list, p, label='BLIMP1', color='black')
    plt.plot(t_list, b, label='BCL-6', color='red')
    plt.plot(t_list, r, label='IRF-4', color='green')
    # plt.plot(t_list_inside, bcr_list)
    # plt.plot(t_list_inside, cd0_list)

    # plt.plot(soln['t'], BCR, label='BCR', color='orange')
    # plt.plot(soln['t'], CD40, label='CD40', color='blue')
    plt.ylabel('Concentration')
    plt.xlabel('Time')
    plt.legend()
    plt.grid()
    plt.title('Concentration over time')
    plt.show()
