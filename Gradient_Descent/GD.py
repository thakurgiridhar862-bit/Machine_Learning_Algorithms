import numpy as np


class GS_reg:
    def __init__(self, lr, ep):
        self.m = 0
        self.b = 0
        self.lr = lr
        self.epochs = ep

    def fit(self, X, Y):

        for i in range(self.epochs):
            ls_b = -2 * np.sum(Y - self.m * X - self.b)
            ls_m = -2 * np.sum((Y - self.m * X - self.b) * X)
            self.b -= self.lr * ls_b
            self.m -= self.lr * ls_m

    def predict(self, X):
        return self.m * X + self.b
