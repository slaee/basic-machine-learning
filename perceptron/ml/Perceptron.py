import numpy as np

from ml.ImageDataset import ImageDataset

imgDatasets = ImageDataset()
imgDatasets.to2Dflatten()

class Perceptron:
    def __init__(self, X=imgDatasets.dataX(), Y=imgDatasets.labelY(), bias=0.2, learning_rate=0.01):
        self.X = X
        self.Y = Y
        self.bias = np.zeros(1, dtype=float)
        self.learning_rate = learning_rate
        self.weights = np.zeros(len(X[0]), dtype=float)
        self.d_total = 0

    def sumNodes(self, X_i):
        return np.dot(X_i, self.weights) + self.bias # net input

    def activation(self, sumNodes):
        return np.where(sumNodes >= 0, 1, 0) # activation function

    def predict(self, X):
        linear_output = self.sumNodes(X)
        y_pred = self.activation(linear_output)
        return y_pred
    
    def train(self):
        zipped_data = list(zip(self.X, self.Y))
        np.random.shuffle(zipped_data)
        self.X, self.Y = zip(*zipped_data)
        epoch = 100
        for i in range(epoch):
            for x_i, y_i in zip(self.X, self.Y):
                linear_output = self.sumNodes(x_i)
                y_pred = self.activation(linear_output)
                error = y_i - y_pred
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error
                self.d_total += np.abs(error)
            print("\n[EPOCH]: ", i)
            print("Weights: ", self.weights)
            print("Total error: ", self.d_total)

    def evaluate(self, X, Y):
        correct = 0
        for x_i, y_i in zip(X, Y):
            linear_output = self.sumNodes(x_i)
            y_pred = self.activation(linear_output)
            if y_pred == y_i:
                correct += 1
        return correct / len(X)

    def update(self, x, y):
        linear_output = self.sumNodes(x)
        y_pred = self.activation(linear_output)
        error = y - y_pred
        self.weights += self.learning_rate * error * x
        self.bias += self.learning_rate * error
        self.d_total += np.abs(error)
        print("\n[UPDATE]")
        print("Weights: ", self.weights)
        print("Error: ", error)
        print("Total error: ", self.d_total)
