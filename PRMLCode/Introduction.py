import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(
    '../PRMLNoteAndimplemented/Utilities/Helper'))
sys.path.insert(0, os.path.abspath(
    '../PRMLNoteAndimplemented/Model/Linear'))
import Polynomial
import RidgeRegression
import LinearRegression
import BayesianRegression

def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t


def func(x):
    return np.sin(2*np.pi*x)


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))

if __name__ == "__main__":
    x_train, y_train = create_toy_data(func, 10, 0.25)
    x_test = np.linspace(0, 1, 100)
    y_test = func(x_test)
    plt.scatter(x_train, y_train, facecolor="none",
                edgecolor="b", s=50, label="training data")
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.legend()
    plt.show()

    for i, degree in enumerate([0, 1, 3, 9]):
        plt.subplot(2, 2, i+1)
        feature = Polynomial.PolynomialFeature(degree)
        x_train_t = feature.transform(x_train)
        x_test_t = feature.transform(x_test)
        model = LinearRegression.LinearRegressionModel()
        model.fit(x_train_t, y_train)
        y = model.predict(x_test_t)
        plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
        plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
        plt.plot(x_test, y, c="r", label="fitting")
        plt.ylim(-1.5, 1.5)
        plt.annotate("M={}".format(degree), xy=(-0.15, 1))
plt.legend(bbox_to_anchor=(1.05, 0.64), loc=2, borderaxespad=0.)
plt.show()

training_errors = []
test_errors = []

for i in range(10):
    feature = Polynomial.PolynomialFeature(i)
    x_train_t = feature.transform(x_train)
    x_test_t = feature.transform(x_test)
    model = LinearRegression.LinearRegressionModel()
    model.fit(x_train_t, y_train)
    y = model.predict(x_test_t)
    training_errors.append(rmse(model.predict(x_train_t), y_train))
    test_errors.append(rmse(model.predict(x_test_t), y_test+np.random.normal(scale=0.25, size=len(y_test))))
plt.plot(training_errors, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
plt.plot(test_errors, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("RMSE")
plt.show()

feature = Polynomial.PolynomialFeature(9)
x_train_t = feature.transform(x_train)
x_test_t = feature.transform(x_test)
model = RidgeRegression.RidgeRegressionModel(alpha=1e-3)
model.fit(x_train_t, y_train)
y = model.predict(x_test_t)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="fitting")
plt.ylim(-1.5, 1.5)
plt.legend()
plt.annotate("M=9", xy=(-0.15, 1))
plt.show()

model = BayesianRegression.BayesianRegressionModel(alpha=2e-3, beta=2)
model.fit(x_train_t, y_train)
y, y_err = model.predict(x_test_t, return_std=True)
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="mean")
plt.fill_between(x_test, y - y_err, y + y_err, color="pink", label="std.", alpha=0.5)
plt.xlim(-0.1, 1.1)
plt.ylim(-1.5, 1.5)
plt.annotate("M=9", xy=(0.8, 1))
plt.legend(bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0.)
plt.show()