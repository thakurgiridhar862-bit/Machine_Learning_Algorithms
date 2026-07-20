import numpy as np


class GS_Reg:
    def __init__(self, lr, ep):

        self.co = None
        self.inter = None
        self.lr = lr
        self.ep = ep

    def fit(self, X_train, Y_train):

        self.inter = 0
        self.co = np.ones(X_train.shape[0])
