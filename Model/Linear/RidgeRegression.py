import numpy as np
import Regression

class RidgeRegressionModel(Regression.RegressionModel):
    def __init__(self, alpha:float=1.):
        self.alpha = alpha

    def fit(self, X:np.ndarray, t:np.ndarray):
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t)

    def predict(self, X:np.ndarray):
        return X @ self.w