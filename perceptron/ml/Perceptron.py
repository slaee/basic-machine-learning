import numpy as np

from ml.ImageDataset import ImageDataset

imgDatasets = ImageDataset()
imgDatasets.to2Dflatten()

class Perceptron:
    def __init__(self, X=imgDatasets.dataX(), Y=imgDatasets.labelY(), bias=0.2, learning_rate=0.01):
        self.X = X
        self.Y = Y
        self.bias = bias
        self.learning_rate = learning_rate
        self.weights = np.random.rand(1225)

    def sumNodes(self, X_i):
        return np.dot(X_i, self.weights) + self.bias

    def activation(self, sumNodes):
        return np.where(sumNodes >= 0, 1, 0)

    def predict(self, X):
        linear_output = self.sumNodes(X)
        y_pred = self.activation(linear_output)
        return y_pred
    
    def update(self):
        zipped_data = list(zip(self.X, self.Y))
        np.random.shuffle(zipped_data)
        self.X, self.Y = zip(*zipped_data)
        d_total = 0
        epoch = 1000
        for i in range(epoch):
            for x_i, y_i in zip(self.X, self.Y):
                linear_output = self.sumNodes(x_i)
                y_pred = self.activation(linear_output)
                error = y_i - y_pred
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error
                d_total += np.abs(error)