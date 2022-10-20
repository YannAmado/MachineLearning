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
            b.append(np.random.normal(loc=0, scale=1, size=layers[l]))
            if (l + 1) < len(layers):
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

def ReLU(n: int):
    return max(0, n)

def ReLU_derivative(n):
    n = np.where(n <= 0, 0, 1)
    return n


def feedForward(net: Network) -> Network:
    for l in range(1, net.nLayers):
        for receivingNeuron in range(net.layers[l]):
            # resetting the z as to not take any info from the activation of the previous number
            net.z[l][receivingNeuron] = 0
            for givingNeuron in range(net.layers[l - 1]):
                net.z[l][receivingNeuron] += net.a[l - 1][givingNeuron] * net.w[l - 1][givingNeuron][receivingNeuron]
            net.z[l][receivingNeuron] += net.b[l][receivingNeuron]
            net.a[l][receivingNeuron] = ReLU(net.z[l][receivingNeuron])

    return net


def setInput(net: Network, MNISTnumber):
    for i in MNISTnumber:
        net.a[0][i] = i
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

        net.b[l - 1] = net.b[l - 1] + learningRate * (nablaB / batchSize)
        net.w[l - 1] = net.w[l - 1] + learningRate * (nablaW / batchSize)

        # finding the error one layer behind
        # in the book it needs a transpose because its weight[layer][receivingNeuron][givingNeuron]
        # but my implementation uses weight[layer][givingNeuron][receivingNeuron] so it's not necessary
        delta = (np.dot(net.w[l - 1], delta)) * ReLU_derivative(net.z[l - 2])


def SGD(net: Network, X: list, y: list, batchSize: int, nEpochs, learningRate):
    for epoch in range(nEpochs):
        delta = 0
        batch = rd.sample(range(len(X)), batchSize)
        for i in batch:
            number = np.squeeze(X[i])
            net = setInput(net, number)
            delta += (net.a[-1] - y[i]) * ReLU_derivative(net.z[-1])

        # taking the average of the results
        delta = delta / batchSize
        net = backProp(net, delta, batchSize, learningRate)
    return net

net = Network([784,30,10])
train_X = mnist.train_images()
train_y = mnist.train_labels()

net = SGD(net, train_X, train_y, 100, 50, 0.05)