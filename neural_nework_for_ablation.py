import numpy as np

# cxs  for ablation study
class TwoLayerNeuralNetwork:
    def __init__(self, input_dim=7, hidden_dim=4, output_dim=1):
        # initialize weights randomly with mean 0
        self.syn0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
        self.syn1 = 2 * np.random.random((hidden_dim, output_dim)) - 1

    # sigmoid function
    def nonlin(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, n_iter=10000):
        for _ in range(n_iter):
            # forward propagation
            l0 = X
            l1 = self.nonlin(np.dot(l0, self.syn0))
            l2 = self.nonlin(np.dot(l1, self.syn1))

            # how much did we miss?
            l2_error = y - l2

            # multiply how much we missed by the slope of the sigmoid at the values in l2
            l2_delta = l2_error * self.nonlin(l2, True)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            l1_error = l2_delta.dot(self.syn1.T)

            # in what direction is the target l1?
            # were we really sure? if so<IPython.core.display.HTML object> don't change too much.
            l1_delta = l1_error * self.nonlin(l1,True)

            # update weights
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += l0.T.dot(l1_delta)

    def predict(self, X):
        # forward propagation
        l0 = X
        l1 = self.nonlin(np.dot(l0, self.syn0))
        l2 = self.nonlin(np.dot(l1, self.syn1))
        return l2

