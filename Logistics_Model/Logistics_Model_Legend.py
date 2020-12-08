import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from sklearn.metrics import mean_squared_error
import warnings


class Logistics():

    def __init__(self, x, y, **kwargs):
        self.x_train = x
        self.y_train = y

        self.best_initials = None
        self.popt = None
        self.pcov = None

        # self.y_max = 1
        # self.normalize()

        self.learn_initial_variables(**kwargs)

    # def normalize(self):
    #     self.y_max = np.max(self.y_train)
    #     self.y_train  = self.y_train / np.max(self.y_train)

    def epidemic_logistic_model(self, t, r0, x0, xm):
        exp = np.exp(r0 * t)
        return (xm * exp * x0) / (xm + (exp - 1) * x0)

    def loss(self, pars):
        warnings.filterwarnings("ignore")  # do not print warnings by genetic algorithm
        y_hat = self.epidemic_logistic_model(self.x_train, *pars)
        # return mean_squared_error(self.y_train, y_hat)
        return np.sum(y_hat-self.y_train)

    def learn_initial_variables(self, default=True, r0_bounds=None, x0_bounds=None, xm_bounds=None):
        if default:
            bounds = [(0, 1), (0, np.max(self.y_train)), (0, 4e6)]
        else:
            bounds = [r0_bounds, x0_bounds, xm_bounds]
        # print(bounds)
        self.best_initials = differential_evolution(self.loss, bounds, seed=10).x

    def fit(self, **kwargs):
        self.popt, self.pcov = curve_fit(self.epidemic_logistic_model, self.x_train, self.y_train, self.best_initials,
                                         bounds=([0, 0, 0], [1, np.inf, np.inf]), **kwargs)
        return self.popt

    def predict(self, x):
        y_hat = self.epidemic_logistic_model(x, *(self.popt))
        return y_hat

    def mse(self, x, y):
        y_hat = self.predict(x)
        return mean_squared_error(y, y_hat)
    
    def predict_increase(self):...