import pandas as pd
import numpy as np
from openpyxl import load_workbook
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import svd
import matplotlib
import matplotlib.pyplot as plt


class MVPR_forward():

    def __init__(self,training_data, training_targets, validation_training,validation_targets):
        self.training_data= training_data
        self.training_targets = training_targets
        self.validation_training = validation_training
        self.validation_targets = validation_targets

    def TSVD_error(self,X_val,V,S,U,ind):

        SIGMA = np.diag(S)

        T_INV =np.matmul( np.matmul(V[:, 0: ind],np.linalg.inv(SIGMA[0:ind, 0:ind] )),np.transpose(U[:, 0: ind]))
        B = np.matmul(T_INV , self.training_targets)

        predicted_validation_profiles = np.matmul(X_val , B)

        error = (np.linalg.norm(self.validation_targets-predicted_validation_profiles)/np.linalg.norm(self.validation_targets))*100
        # print(error)
        return error, B

    def GSS(self,X_train,X_val):

        U, S, VT = svd(X_train)

        V =  np.transpose(VT)

        ind_low_1 = 1
        ind_high_1 = len(S)

        F_ind_low_1, CM = self.TSVD_error(X_val, V, S, U, ind_low_1)
        F_ind_high_1, CM = self.TSVD_error(X_val, V, S, U, ind_high_1)

        # F_ind_lower = [F_ind_low_1]
        # F_ind_higher = [F_ind_high_1]
        # ind_lower = [ind_low_1]
        # ind_higher = [ind_high_1]
        LL = 1

        while LL < 20:

            distance = 0.61803398875 * (np.log10(ind_high_1) - np.log10(ind_low_1))
            ind_low_2 = round(10 ** (np.log10(ind_high_1) - distance))
            ind_high_2 = round(10 ** (np.log10(ind_low_1) + distance))


            F_ind_low_2, CM = self.TSVD_error(X_val, V, S, U, ind_low_2)

            F_ind_high_2, CM = self.TSVD_error(X_val, V, S, U, ind_high_2)

            if F_ind_low_2 > F_ind_high_2:
                ind_low_1 = ind_low_2
                # [F_ind_low_1] = F_ind_low_2
                # F_ind_lower = [F_ind_lower F_ind_low_1]
                # ind_lower = [ind_lower ind_low_1]


            if F_ind_low_2 < F_ind_high_2:
                ind_high_1 = ind_high_2
                # [F_ind_high_1] = F_ind_high_2
                # F_ind_higher = [F_ind_higher F_ind_high_1]
                # ind_higher = [ind_higher ind_high_1]



            LL = LL + 1




        ind = ((ind_low_1 + ind_high_1) / 2)
        error, CM = self.TSVD_error(X_val, V, S, U, int(ind))

        return error, CM


    def find_order(self):

        order = np.arange(1,7,1)
        error =np.zeros(len(order))
        for i in range(0,len(order),1):
            poly = PolynomialFeatures(degree=order[i])
            X_train=poly.fit_transform(self.training_data)
            X_val = poly.fit_transform(self.validation_training)
            error[i], dummy=self.GSS(X_train,X_val)
            print('order = ', order[i], '| error = ', error[i])
        optimum_order_index = np.where(np.min(error) == error)
        optimum_order = int(order[optimum_order_index])
        return optimum_order

    def compute_CM(self, order):
        poly = PolynomialFeatures(degree=order)
        X_train = poly.fit_transform(self.training_data)
        X_val = poly.fit_transform(self.validation_training)
        error, CM =  self.GSS(X_train, X_val)

        return CM

    def compute(self, CM, order, data):
        poly = PolynomialFeatures(degree=order)
        X = poly.fit_transform(data)

        return np.matmul(X, CM)

