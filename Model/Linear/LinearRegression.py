import numpy as np
import Regression

class LinearRegressionModel(Regression.RegressionModel):
    def fit(self, X: np.ndarray, t: np.ndarray):
        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t))

    def predict(self, X:np.ndarray, return_std:bool=False):
        y = X @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y