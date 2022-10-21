import numpy as np
import random as rd
import mnist


class Network:
    def __init__(self, layers: list):
        b = []
        w = []
        a = []
        z = []
        for l in range(0, len(layers)):
            # skipping one layer for the weights and biases
            if (l + 1) < len(layers):
                b.append(np.random.normal(loc=0, scale=1, size=layers[l + 1]))
                w.append(np.random.normal(loc=0, scale=3, size=[layers[l], layers[l + 1]]))
            a.append(np.zeros(layers[l]))
            z.append(np.zeros(layers[l]))

        # b[i][j] -> i is which layer, j which neuron
        # w[i][j][k] -> i is which layer, j which neuron of the first layer, k which neuron of the second layer
        self.b = b
        self.w = w
        self.a = a
        self.z = z
        self.nLayers = len(layers)
        self.layers = layers

def sigmoid(n: float):
    return 1.0/(1.0+np.exp(-n))

def sigmoid_derivative(n: float):
    """Derivative of the sigmoid function."""
    return sigmoid(n)*(1-sigmoid(n))



def feedForward(net: Network) -> Network:
    for l in range(0, net.nLayers - 1):
        for receivingNeuron in range(net.layers[l + 1]):
            # resetting the z as to not take any info from the activation of the previous number
            net.z[l + 1][receivingNeuron] = 0
            for givingNeuron in range(net.layers[l]):
                net.z[l + 1][receivingNeuron] += net.a[l][givingNeuron] * net.w[l][givingNeuron][receivingNeuron]
            net.z[l + 1][receivingNeuron] += net.b[l][receivingNeuron]
            net.a[l + 1][receivingNeuron] = sigmoid(net.z[l + 1][receivingNeuron])

    return net


def setInput(net: Network, MNISTnumber):
    numberArr = np.asarray(MNISTnumber).flatten()
    for i in range(net.layers[0]):
        net.a[0][i] = numberArr[i]
    net = feedForward(net)

    return net


def backProp(net: Network, delta, batchSize, learningRate) -> Network:
    layers = net.layers
    for l in range(net.nLayers - 1, 0, -1):
        nablaB = delta

        # not too sure about nablaW
        nablaW = np.zeros([layers[l - 1], layers[l]])
        for j in range(layers[l]):
            for k in range(layers[l - 1]):
                nablaW[k][j] += net.a[l - 1][k] * delta[j]

        net.b[l - 1] = net.b[l - 1] - learningRate * (nablaB / batchSize)
        net.w[l - 1] = net.w[l - 1] - learningRate * (nablaW / batchSize)

        # finding the error one layer behind
        # in the book it needs a transpose because its weight[layer][receivingNeuron][givingNeuron]
        # but my implementation uses weight[layer][givingNeuron][receivingNeuron] so it's not necessary
        if l >= 0:
            delta = (np.dot(net.w[l - 1], delta)) * sigmoid_derivative(net.z[l - 1])

    return net


def SGD(net: Network, X: list, y: list, batchSize: int, nEpochs: int, learningRate) -> Network:
    for epoch in range(nEpochs):
        print(epoch)
        delta = 0
        batch = rd.sample(range(len(X)), batchSize)
        for i in batch:
            net = setInput(net, X[i])
            # not too sure about the meaning of the y in the equation (a^L_j - y_j)
            for j in range(net.layers[-1]):
                if y[i] == j:
                    delta += (net.a[-1][j] - 1) * sigmoid_derivative(net.z[-1])
                else:
                    delta += (net.a[-1][j] - 0) * sigmoid_derivative(net.z[-1])

        # taking the average of the results
        delta = delta / batchSize
        net = backProp(net, delta, batchSize, learningRate)
    return net


net = Network([784,30,10])
train_X = mnist.train_images()
train_y = mnist.train_labels()

net = SGD(net, train_X, train_y, 100, 50, 0.05)