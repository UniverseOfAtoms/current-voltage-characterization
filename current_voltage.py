import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


class Fitting:
    def __init__(self, dataframes):
        self.tables = dataframes
        self.model = LinearRegression()
        self.beg_of_lin_area = [0.92, 0.9, 0.74, 0.68, 0.6, 0.54, 0.48]
        self.end_of_lin_area = [1.12, 1.16, 1.08, 1.02, 0.98, 0.96, 0.92]
        self.temp = [187, 200, 250, 288, 300, 321, 350]
        self.type_params = {'Is': float, 'n': float, 'Temperature': int, 'ln(Is/T2)': float, '1/kBT': float}
        self.params = pd.DataFrame(columns=['Is', 'n', 'Temperature', 'ln(Is/T2)', '1/kBT']).astype(self.type_params)

    def new_variables(self):
        for i, df in enumerate(self.tables):
            df["lnI"] = df["Anode I"].apply(lambda x: math.log(x))
            df["Temperature"] = self.temp[i]

    def plot_ln(self):
        self.new_variables()
        fig, ax = plt.subplots()
        for df in self.tables:
            ax.plot(df["Anode V"], df["lnI"], label=f"{df['Temperature'][0]}K")
        ax.tick_params(direction='in', right=True, left=True, bottom=True, top=True)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("lnI (A)")
        ax.set_xlim(-0.5, 2.5)
        ax.legend()
        plt.show()

    def lin_fit(self):
        self.new_variables()
        data = []
        for i, df in enumerate(self.tables):
            x_new = df.loc[(df["Anode V"] >= self.beg_of_lin_area[i]) & (df["Anode V"] <= self.end_of_lin_area[i]),
                           "Anode V"]
            y_new = df.loc[(df["Anode V"] >= self.beg_of_lin_area[i]) & (df["Anode V"] <= self.end_of_lin_area[i]),
                           "lnI"]
            # Linear regression and calculation of r2
            self.model.fit(x_new.values.reshape(-1, 1), y_new)
            r2 = self.model.score(x_new.values.reshape(-1, 1), y_new)
            #print(r2)

            # Plotting data and fitted line
            #plt.scatter(df["Anode V"], df["lnI"])
            #plt.plot(x_new, self.model.intercept_+self.model.coef_*x_new, color='r')
            #plt.xlabel("Voltage (V)")
            #plt.ylabel("lnI (A)")
            #plt.show()

            # Calculate Sum Squared Error SSE and degrees of freedom
            y_pred = self.model.predict(x_new.values.reshape(-1, 1))
            SSE = np.sum((y_new - y_pred)**2)
            df = len(x_new) - 2

            # Calculate mean squared error (MSE)
            MSE = SSE / df

            # Calculate sum of squares of x
            SSx = np.sum((x_new - np.mean(x_new)) ** 2)

            # Calculate standard error of intercept and slope
            se_intercept = np.sqrt(MSE*(1/(len(x_new))+(np.mean(x_new)**2)/ SSx))
            se_slope = np.sqrt(MSE / SSx)

            d = {'Intercept': self.model.intercept_, 'Standard_Error_Intercept': se_intercept,
                 'Slope': self.model.coef_[0], 'Standard_Error_Slope': se_slope,
                 'Temperature': self.temp[i], 'R-Square': r2}
            data.append(d)
            int_slope_type = {'Intercept': float, 'Standard_Error_Intercept': float,
                              'Slope': float, 'Standard_Error_Slope': float,
                              'Temperature': float, 'R-Square': float
                              }
            index = pd.Index(range(len(data)))
            int_slope = pd.DataFrame(data, index=index).astype(int_slope_type)
            n = 1.60E-19/(1.38e-23*self.temp[i]*self.model.coef_[0])
            Is = np.exp(self.model.intercept_)
            IsT2 = np.log(Is/self.temp[i]**2)
            kBT = (1/(1.3e-23*self.temp[i]))*1.6e-19

            self.params.loc[i] = [Is, n, self.temp[i], IsT2, kBT]
        return self.params

    def rich_graph(self):
        self.lin_fit()
        self.model.fit(self.params['1/kBT'].values.reshape(-1, 1), self.params['ln(Is/T2)'].values.reshape(-1, 1))
        intercept = self.model.intercept_[0]
        slope = self.model.coef_[0][0]

        # Calculating r2
        r2 = self.model.score(self.params['1/kBT'].values.reshape(-1, 1), self.params['ln(Is/T2)'].values.reshape(-1, 1))
        #print(r2)

        # Calculate Sum Squared Error SSE and degrees of freedom
        y_pred = self.model.predict(self.params['1/kBT'].values.reshape(-1, 1))
        SSE = np.sum((self.params['ln(Is/T2)'].values.reshape(-1, 1) - y_pred) ** 2)
        df = len(self.params['1/kBT'].values.reshape(-1, 1)) - 2
        # Calculate mean squared error (MSE)
        MSE = SSE / df
        # Calculate mean squared error (MSE)
        MSE = SSE / df
        # Calculate sum of squares of x
        SSx = np.sum((self.params['1/kBT'].values.reshape(-1, 1) - np.mean(self.params['1/kBT'].values.reshape(-1, 1))) ** 2)
        # Calculate standard error of intercept and slope
        se_intercept = np.sqrt(MSE * (1 / (len(self.params['1/kBT'].values.reshape(-1, 1))) + (np.mean(self.params['1/kBT'].values.reshape(-1, 1)) ** 2) / SSx))
        se_slope = np.sqrt(MSE / SSx)
        #print(f"Standard error of slope: {se_slope}, and standard error of intercept: {se_intercept}")
        rich_const = np.exp(intercept)/1e-2
        for index, row in self.params.iterrows():
            barrier_height = ((-1.3e-23*slope*row['Temperature'])/1.6e-19)*6.624e18
            #print(barrier_height)

        # Plotting Richardson's graph
        plt.scatter(self.params['1/kBT'], self.params['ln(Is/T2)'])
        plt.plot(self.params['1/kBT'].values.reshape(-1, 1),
                 self.model.intercept_ + self.model.coef_ * self.params['1/kBT'].values.reshape(-1, 1), color='r')
        plt.tick_params(direction='in', right=True, left=True, bottom=True, top=True)
        plt.figtext(0.54, 0.74, f"A*= {rich_const}AK-2cm-2", fontsize=8, color='black')
        plt.xlabel("1/kBT")
        plt.ylabel("ln(Is/T2)")
        plt.show()


