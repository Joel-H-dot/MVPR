import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import svd

class MVPR_forward():

    def __init__(self,training_data, training_targets, validation_data,validation_targets, regularisation = 'TSVD', verbose=False, search='exponent'):
        self.training_data= training_data
        self.training_targets = training_targets
        self.validation_data = validation_data
        self.validation_targets = validation_targets
        self.verbose = verbose
        self.search = search
        self.regularisation = regularisation
        self.reg_type = 'Identity'
        self.tik_lam_upper = 10e3
        self.tik_lam_lower = 10e-10



    def TSVD_error(self,X_val,V,S,U,ind):

        SIGMA = np.diag(S)

        T_INV =np.matmul( np.matmul(V[:, 0: ind],np.linalg.inv(SIGMA[0:ind, 0:ind] )),np.transpose(U[:, 0: ind]))
        B = np.matmul(T_INV , self.training_targets)

        predicted_validation_profiles = np.matmul(X_val , B)

        error = (np.linalg.norm(self.validation_targets - predicted_validation_profiles, axis=1) / np.linalg.norm(
            self.validation_targets, axis=1)) * 100
        error = np.sum(error) / len(error)
        return error, B


    def GSS_TSVD(self,X_train,X_val):

        number_of_elements = len(X_train[0, :]) * len(X_train[:, 0])

        if number_of_elements > 2 ** 26:
            raise Exception("Matrix too large, SVD will crash using scipy. Limit on indexing with lapack package.") # see https://github.com/scipy/scipy/issues/14001

        U, S, VT = svd(X_train)

        V =  np.transpose(VT)

        ind_low_1 = 1
        ind_high_1 = len(S)

        F_ind_low_1, CM = self.TSVD_error(X_val, V, S, U, ind_low_1)
        F_ind_high_1, CM = self.TSVD_error(X_val, V, S, U, ind_high_1)

        LL = 1

        while LL < 20:

            if self.verbose:
                print('       Truncation point Low = ', ind_low_1, ', Cross validation MPEN = ', F_ind_low_1,
                      ' || Truncation point High = ', ind_high_1, ', Cross validation MPEN = ', F_ind_high_1)

            if self.search == 'exponent':
                distance = 0.61803398875 * (np.log10(ind_high_1) - np.log10(ind_low_1))
                ind_low_2 = round(10 ** (np.log10(ind_high_1) - distance))
                ind_high_2 = round(10 ** (np.log10(ind_low_1) + distance))
            else:
                distance = 0.61803398875 * (ind_high_1 - ind_low_1)
                ind_low_2 = round(ind_high_1 - distance)
                ind_high_2 = round(ind_low_1 + distance)


            F_ind_low_2, CM = self.TSVD_error(X_val, V, S, U, ind_low_2)

            F_ind_high_2, CM = self.TSVD_error(X_val, V, S, U, ind_high_2)

            if F_ind_low_2 > F_ind_high_2:
                ind_low_1 = ind_low_2
                F_ind_low_1 = F_ind_low_2



            if F_ind_low_2 < F_ind_high_2:
                ind_high_1 = ind_high_2
                F_ind_high_1 = F_ind_high_2

            LL = LL + 1


        if self.verbose:
            print('       Truncation point Low = ', ind_low_1, ', Cross validation MPEN = ', F_ind_low_1,
                  ' || Truncation point High = ', ind_high_1, ', Cross validation MPEN = ', F_ind_high_1)

        if self.search == 'exponent':
            ind_exp = (np.log10(ind_high_1) + np.log10(ind_low_1))/2
            ind = round(10**ind_exp)
        else:
            ind = round((ind_low_1 + ind_high_1) / 2)

        error, CM = self.TSVD_error(X_val, V, S, U, int(ind))

        return error, CM

    def Tik_error(self, X_val, X_train, lambda_param):


        XTX = np.matmul(np.transpose(X_train),X_train)
        XTA = np.matmul(np.transpose(X_train), self.training_targets)

        prior = np.matmul(np.linalg.inv(XTX),XTA)

        if self.reg_type == 'Identity':
            RM = np.ones((len(prior[:,0]),len(prior[:,0])))
            RM = np.diag(np.diag(RM))
        elif self.reg_type == 'Finite Difference':
            RM = np.zeros((len(prior[:,0]),len(prior[:,0])))

            for i in range(0, 9):
                RM[i, i + 1] = 0.5
            for i in range(0, 9):
                RM[i + 1, i] = -0.5
            RM[0, 0] = -1
            RM[0, 1] = 1
            RM[-1, -1] = 1
            RM[-1, -2] = -1
        else:
            RM = np.zeros((len(prior[:,0]),len(prior[:,0])))


        RMTRM= np.matmul(np.transpose(RM),RM)

        denom = XTX+lambda_param*RMTRM
        num = XTA +lambda_param*np.matmul(RMTRM,prior)


        B = np.matmul(np.linalg.inv(denom),num)


        predicted_validation_profiles = np.matmul(X_val, B)

        error = (np.linalg.norm(self.validation_targets - predicted_validation_profiles,axis=1) / np.linalg.norm(
            self.validation_targets,axis=1)) * 100
        error = np.sum(error)/len(error)


        return error, B

    def GSS_Tik(self, X_train, X_val):

        iterations = 20

        param_low_1 = self.tik_lam_lower
        param_high_1 = self.tik_lam_upper

        F_ind_low_1, CM = self.Tik_error(X_val, X_train, param_low_1)
        F_ind_high_1, CM = self.Tik_error(X_val, X_train, param_high_1)

        LL = 0

        F_low = F_ind_low_1
        F_high = F_ind_high_1
        param_low = param_low_1
        param_high = param_high_1

        while LL < iterations:

            if self.verbose:
                print('       Reg Low = ', param_low_1, ', Cross validation MPEN = ', F_ind_low_1,
                      ' || Reg High = ', param_high_1, ', Cross validation MPEN = ', F_ind_high_1)

            if self.search == 'exponent':
                distance = 0.61803398875 * (np.log10(param_high_1) - np.log10(param_low_1))
                param_low_2 = (10 ** (np.log10(param_high_1) - distance))
                param_high_2 = (10 ** (np.log10(param_low_1) + distance))
            else:
                distance = 0.61803398875 * (param_high_1 - param_low_1)
                param_low_2 = (param_high_1 - distance)
                param_high_2 = (param_low_1 + distance)

            F_ind_low_2, CM = self.Tik_error(X_val, X_train, param_low_2)

            F_ind_high_2, CM = self.Tik_error(X_val, X_train, param_high_2)

            if F_ind_low_2 > F_ind_high_2:
                param_low_1 = param_low_2
                F_ind_low_1 = F_ind_low_2

                F_low = np.append(F_low,F_ind_low_1)
                param_low =  np.append(param_low,param_low_1)


            if F_ind_low_2 < F_ind_high_2:
                param_high_1 = param_high_2
                F_ind_high_1 = F_ind_high_2

                F_high = np.append(F_high, F_ind_high_1)
                param_high = np.append(param_high, param_high_1)

            LL = LL + 1


        if self.verbose:
            print('       Reg Low = ', param_low_1, ', Cross validation MPEN = ', F_ind_low_1,
                  ' || Reg High = ', param_high_1, ', Cross validation MPEN = ', F_ind_high_1)

        if self.search == 'exponent':
            param_exp = (np.log10(param_high_1) + np.log10(param_low_1)) / 2
            param = (10 ** param_exp)
        else:
            param = ((param_low_1 + param_high_1) / 2)

        error, CM = self.Tik_error(X_val, X_train, param)

        return error, CM

    def select_func(self):
        if self.regularisation == 'TSVD':
            self.GSS = self.GSS_TSVD
        else:
            self.GSS = self.GSS_Tik

    def find_order(self):

        self.select_func()

        order = np.arange(1,7,1)
        error =np.zeros(len(order))
        for i in range(0,len(order),1):

            poly = PolynomialFeatures(degree=order[i])

            X_train=poly.fit_transform(self.training_data)
            X_val = poly.fit_transform(self.validation_data)

            try:
                error[i], dummy=self.GSS(X_train,X_val)
            except:
                error[i]=np.inf

            if self.verbose:
                print('order = ', order[i], '| error = ', error[i])
                
        optimum_order_index = np.where(np.min(error) == error)
        optimum_order = int(order[optimum_order_index])
        return optimum_order

    def compute_CM(self, order):
        self.select_func()
        poly = PolynomialFeatures(degree=order)
        X_train = poly.fit_transform(self.training_data)
        X_val = poly.fit_transform(self.validation_data)
        error, CM =  self.GSS(X_train, X_val)

        return CM

    def compute(self, CM, order, data):
        poly = PolynomialFeatures(degree=order)
        X = poly.fit_transform(data)

        return np.matmul(X, CM)

