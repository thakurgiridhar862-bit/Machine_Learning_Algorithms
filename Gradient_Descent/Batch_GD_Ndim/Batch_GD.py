import numpy as np


class GS_Reg:
    def __init__(self, lr, ep):

        self.co = None
        self.inter = None
        self.lr = lr
        self.ep = ep

    def fit(self, X_train, Y_train):

        self.inter = 0
        self.co = np.ones(X_train.shape[1])

        for i in range(self.ep):
            y_hat = np.dot(X_train, self.co) + self.inter
            in_der = -2 * np.mean(Y_train - y_hat)
            self.inter -= self.lr * in_der

            co_der = (-2 / X_train.shape[0]) * np.dot(X_train.T, Y_train - y_hat)
            self.co -= self.lr * co_der

    def predict(self, X_test):
        return np.dot(X_test, self.co) + self.inter
