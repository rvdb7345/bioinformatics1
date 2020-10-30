import pandas as pd
import numpy as np
import csv
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

class full_model(object):
    """ A class containing the parameters, equations and necessary functions for the standard Martinez model """

    def __init__(self, name, CD40, mu_r, sigma_r, k_r, l_r, mu_b, sigma_b, k_b, l_b, mu_p, sigma_p, k_p, l_p):
        self.name = name
        
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

        self.fitness = self.calc_fitness()

    def equation_irf(self, r):        
        drdt = (self.mu_r + self.sigma_r * r ** 2 / (self.k_r ** 2 + r ** 2) + self.CD40 - self.l_r * r)

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

    def plot(self):
        r_list = np.arange(0, 12, 0.001)
        drdt = self.equation_irf(r_list)
        dbdt_GC = self.equation_bcl_GC(r_list)
        dbdt_PC = self.equation_bcl_PC(r_list)
        dpdt_GC = self.equation_blimp_GC(r_list)
        dpdt_PC = self.equation_blimp_PC(r_list)

        fig = plt.figure()
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.plot(r_list, drdt, label='fit', color='green')
        plt.scatter(inter1_data_IRF, np.zeros(len(inter1_data_IRF)), label='GC')
        plt.scatter(inter2_data_IRF, np.zeros(len(inter2_data_IRF)), label='PC')
        plt.scatter(self.intersections_irf.x, [0, 0, 0], marker='d', color='black', label="roots", zorder=10)
        # plt.ylim(-2.5, 2.5)
        # plt.xlim(3, 12)
        plt.xlabel('[IRF4] ($10^{-8}$ M)', fontsize=14)
        plt.ylabel('dr/dt ($10^{-8}$ M / 4h)', fontsize=14)
        plt.legend(fontsize=14)
        fig.savefig("PlotsReport/{}_IRF4_fitness:{:.4f}.png".format(self.name.capitalize(), self.fitness))
        plt.close(fig)

        if "IRF4" in self.name: return

        fig = plt.figure()
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.plot(r_list, dbdt_GC, label='fit GC')
        plt.plot(r_list, dbdt_PC, label='fit PC')
        plt.scatter(inter1_data_BCL, np.zeros(len(inter1_data_BCL)), label='GC')
        plt.scatter(inter2_data_BCL, np.zeros(len(inter2_data_BCL)), label='PC')
        plt.scatter([self.intersections_bcl_GC.x, self.intersections_bcl_PC.x], [0, 0], marker='d', color='black', label="roots", zorder=10)
        # plt.ylim(-0.05 * 10 ** -8, 0.09 * 10 ** (-8))
        # plt.xlim(3, 12)
        plt.xlabel('[BCL6] ($10^{-8}$ M)', fontsize=14)
        plt.ylabel('db/dt ($10^{-8}$ M / 4h)', fontsize=14)
        plt.legend(fontsize=14)
        fig.savefig("PlotsReport/{}_BCL_fitness:{:.4f}.png".format(self.name.capitalize(), self.fitness))
        plt.close(fig)

        fig = plt.figure()
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.plot(r_list, dpdt_GC, label='fit GC')
        plt.plot(r_list, dpdt_PC, label='fit PC')
        plt.scatter(inter1_data_BLIMP, np.zeros(len(inter1_data_BLIMP)), label='GC')
        plt.scatter(inter2_data_BLIMP, np.zeros(len(inter2_data_BLIMP)), label='PC')
        plt.scatter([self.intersections_blimp_GC.x, self.intersections_blimp_PC.x], [0, 0], marker='d', color='black', label="roots", zorder=10)
        # plt.ylim(-0.05 * 10 ** -8, 0.09 * 10 ** (-8))
        # plt.xlim(3, 12)
        plt.xlabel('[BLIMP1] ($10^{-8}$ M)', fontsize=14)
        plt.ylabel('dp/dt ($10^{-8}$ M / 4 h)', fontsize=14)
        plt.legend(fontsize=14)
        fig.savefig("PlotsReport/{}_BLIMP_fitness:{:.4f}.png".format(self.name.capitalize(), self.fitness))
        plt.close(fig)

    def calc_fitness(self):
        '''
        Calculates the fitness of an individual as the mse on the time series
        :param ind: an array containing the parameters we want to be fitting
        :return: the fitness of an individual, the lower the better tho
        '''

        # model_ind = full_model(*ind)
        intersections = self.calc_zeropoints()

        if "IRF4" in self.name:
            return abs(sum((min(intersections[0]) - inter1_data_IRF)/inter1_data_IRF)) / len(inter1_data_IRF) + \
                abs(sum((max(intersections[0]) - inter2_data_IRF)/inter2_data_IRF)) / len(inter2_data_IRF)

        beta = (self.mu_r + self.CD40 + self.sigma_r) / (self.l_r * self.k_r)
        p = - self.sigma_r / (self.l_r * self.k_r) + beta

        # instantiate an individual
        if (beta ** 2 > 3) and (beta ** 3 + (beta ** 2 - 3) ** (3 / 2) - 9 * beta / 2 > - 27 / 2 * p) and \
                (beta ** 3 - (beta ** 2 - 3) ** (3 / 2) - 9 * beta / 2 < - 27 / 2 * p) and (beta > 0) and (p > 0):

            
            # NORMALISED FITNESS VERION
            return abs(sum((min(intersections[0]) - inter1_data_IRF)/inter1_data_IRF)) / len(inter1_data_IRF) + \
                abs(sum((max(intersections[0]) - inter2_data_IRF)/inter2_data_IRF)) / len(inter2_data_IRF) + \
                abs(sum((max(intersections[1]) - inter1_data_BCL)/inter1_data_BCL)) / len(inter1_data_BCL) + \
                abs(sum((min(intersections[1]) - inter2_data_BCL)/inter2_data_BCL)) / len(inter2_data_BCL) + \
                abs(sum((min(intersections[2]) - inter1_data_BLIMP)/inter1_data_BLIMP)) / len(inter1_data_BLIMP) + \
                abs(sum((max(intersections[2]) - inter2_data_BLIMP)/inter2_data_BLIMP)) / len(inter2_data_BLIMP)

        else:
            return 10000000


if __name__ == '__main__':

    # name = "IRF4_MartParams_MartData"
    # name = "IRF4_MartParams_WesData"
    # name = "IRF4fit_MuSigma_WesData"
    # name = "IRF4fit_Full_WesData"
    # name = "ModFit_MuSigma_WesData"
    # name = "MartParams_MartData"
    name = "ModFit_MuSigma_MartDataDiv4"
    # name = "FullModFit_MartDataDiv4"
    # name = "FullModFit_WesData"



    if "MartData" in name:
        affymetrix_df = pd.read_csv('matrinez_data.csv') 
    else:
        affymetrix_df = pd.read_csv('wesenhagen_data.csv')
    
    affymetrix_df = affymetrix_df.set_index('Sample')

    if "Div4" in name:
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

    # Standard settings (MartParams), can be overwritten
    CD40 = 0.0001
    mu_p = 10**(-6)
    mu_b = 2
    mu_r = 0.1
    sigma_p = 9
    sigma_b = 100
    sigma_r = 2.6
    k_p = 1
    k_b = 1
    k_r = 1
    lambda_p = 1
    lambda_b = 1
    lambda_r = 1

    if name == "IRF4fit_MuSigma_WesData":
        mu_r = 0.0256
        sigma_r = 9.7883

    if name == "IRF4fit_Full_WesData":
        mu_r = 24.0
        sigma_r = 204.3
        k_r = 12.43
        lambda_r = 10.45
        CD40 = 0.054

    if name == "ModFit_MuSigma_WesData":
        CD40 = 0.00063847
        # Basal transcription rate
        mu_p = 2.247
        mu_b = 7.369
        mu_r = 0.02499
        # Maximum induced transcription rate
        sigma_p = 9.402
        sigma_b = 172.7
        sigma_r = 9.761

    if name == "ModFit_MuSigma_MartDataDiv4":
        CD40 = 0.000526598
        # Basal transcription rate
        mu_p = 8.758e-9
        mu_b = 0.001303
        mu_r = 0.1589
        # Maximum induced transcription rate
        sigma_p = 1.199
        sigma_b = 41.93
        sigma_r = 1.748

    if name == "FullModFit_MartDataDiv4":
        CD40 = 8.910e-5
        # Basal transcription rate
        mu_p = 1.636e-6
        mu_b = 0.2069
        mu_r = 0.3184
        # Maximum induced transcription rate
        sigma_p = 3.346
        sigma_b = 76.74
        sigma_r = 3.016
        # Dissociation constant
        k_p = 0.4515
        k_b = 0.6528
        k_r = 1.155
        # Degradation rate
        lambda_p = 1.821
        lambda_b = 1.597
        lambda_r = 1.600
    
    if name == "FullModFit_WesData":
        CD40 = 0.00047
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



    model = full_model(name, CD40, mu_r, sigma_r, k_r, lambda_r, mu_b, sigma_b, k_b, lambda_b, mu_p, sigma_p, k_p, lambda_p)

    print("Name    = ", name)
    print("Fitness = ", model.fitness)

    model.plot()