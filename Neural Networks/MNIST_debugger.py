import numpy as np
import random as rd


def sigmoid(n: float):
    return 1.0 / (1.0 + np.exp(-n))


def sigmoid_derivative(n: float):
    """Derivative of the sigmoid function."""
    return sigmoid(n) * (1 - sigmoid(n))


def tanh(n: float):
    return np.tanh(n)

def padding(X):
    maxLen = len(max(X, key=len))
    for i in range(len(X)):
        X[i] = np.pad(X[i], (0 ,maxLen - len(X[i])), 'constant', constant_values=(0, 0))
    return X

maxLen = 500
train_X = np.load('train_X.npy', allow_pickle=True)
train_y = np.load('train_y.npy', allow_pickle=True)


def testNetwork(net, test_X, test_y, nTests: int):
    """
    A function to test our network

    It returns the overall accuracy and the numbers our network guessed
    """

    correctOutput = 0
    X = test_X[:nTests]
    y = test_y[:nTests]
    outputs = np.zeros(10)
    for i in range(nTests):
        net.setInput(X[i])
        networkOutput = np.argmax(net.a[-1])
        outputs[networkOutput] += 1
        # print(f"number: {y[i]}, networkOutput: {networkOutput}, activations: {net.a[-1]}")
        if y[i] == networkOutput:
            correctOutput += 1
    acc = correctOutput / nTests
    return acc, outputs


class LSTM:
    """
    h = array of outputs
    x = array of inputs
    """

    def __init__(self, nInputs, nFeatures, nCells, nOutputs, batchSize):
        nGates = 4
        if nCells < nOutputs:
            print('the number of cells cannot be less than the number of outputs')

        # [t][i][j], t = timestep, i = which batch, j = which cell or feature
        x = np.zeros((nInputs, batchSize, nFeatures))
        h = np.zeros((nInputs, batchSize, nCells))
        i = np.zeros((nInputs, batchSize, nCells))
        f = np.zeros((nInputs, batchSize, nCells))
        o = np.zeros((nInputs, batchSize, nCells))
        tildeC = np.zeros((nInputs, batchSize, nCells))
        C = np.zeros((nInputs, batchSize, nCells))
        wxScale = 1 / np.sqrt(nFeatures * nCells)
        whScale = 1 / np.sqrt(nCells * nCells)
        Wxi = np.random.normal(loc=0, scale=wxScale, size=[nFeatures, nCells])
        Wxf = np.random.normal(loc=0, scale=wxScale, size=[nFeatures, nCells])
        Wxc = np.random.normal(loc=0, scale=wxScale, size=[nFeatures, nCells])
        Wxo = np.random.normal(loc=0, scale=wxScale, size=[nFeatures, nCells])
        Whi = np.random.normal(loc=0, scale=wxScale, size=[nCells, nCells])
        Whf = np.random.normal(loc=0, scale=wxScale, size=[nCells, nCells])
        Whc = np.random.normal(loc=0, scale=wxScale, size=[nCells, nCells])
        Who = np.random.normal(loc=0, scale=wxScale, size=[nCells, nCells])
        bi = np.random.normal(loc=0, scale=1, size=[nCells])
        bf = np.random.normal(loc=0, scale=1, size=[nCells])
        bc = np.random.normal(loc=0, scale=1, size=[nCells])
        bo = np.random.normal(loc=0, scale=1, size=[nCells])
        self.x = x
        self.h = h
        self.i = i
        self.f = f
        self.o = o
        self.tildeC = tildeC
        self.C = C
        self.Wxi = Wxi
        self.Wxf = Wxf
        self.Wxc = Wxc
        self.Wxo = Wxo
        self.Whi = Whi
        self.Whf = Whf
        self.Whc = Whc
        self.Who = Who
        self.bi = bi
        self.bf = bf
        self.bc = bc
        self.bo = bo
        self.nCells = nCells
        self.nGates = nGates
        self.nInputs = nInputs

    # the np.array(a) and np.array(b) can be removed after batchSize is implemented

    def forgetGate(self, t):
        if t == 0:
            self.f[t] = sigmoid(np.dot(self.x[t], self.Wxf) + self.bf)
        else:
            self.f[t] = sigmoid(np.dot(self.h[t - 1], self.Whf) + np.dot(self.x[t], self.Wxf) + self.bf)
        return self.f[t]

    def inputGate(self, t):
        if t == 0:
            self.i[t] = sigmoid(np.dot(self.x[t], self.Wxi) + self.bi)
            self.tildeC[t] = tanh(np.dot(self.x[t], self.Wxc) + self.bc)
        else:
            self.i[t] = sigmoid(np.dot(self.h[t - 1], self.Whi) + np.dot(self.x[t], self.Wxi) + self.bi)
            self.tildeC[t] = tanh(np.dot(self.h[t - 1], self.Whc) + np.dot(self.x[t], self.Wxc) + self.bc)
        return self.i[t] * self.tildeC[t]

    def outputGate(self, t, newC):
        if t == 0:
            self.o[t] = sigmoid(np.dot(self.x[t], self.Wxo) + self.bo)
        else:
            self.o[t] = sigmoid(np.dot(self.h[t - 1], self.Who) + np.dot(self.x[t], self.Wxo) + self.bo)
        self.h[t] = self.o[t] * tanh(newC)
        return self.h[t]

    def getNewState(self, t, xElement):
        newC = self.C[t - 1] * self.forgetGate(t)
        newC = newC + self.inputGate(t)
        newH = self.outputGate(t, newC)
        return newC, newH

    def setInput(self, x):
        C = 0
        self.x[0] = x[0]
        self.C[0], self.h[0] = self.getNewState(0, x[0])
        for t in range(1, self.nCells):
            self.x[t] = x[t]
            self.C[t], self.h[t] = self.getNewState(t, x[t])
        return

    def backProp(self, y):
        deltaOut = 0
        # a = C~, state = C
        i = self.i
        f = self.f
        C = self.C
        o = self.o
        deltaGates = np.zeros([self.nCells, self.nGates])
        deltaState = 0
        for t in range(self.nCells, 0, -1):
            delta = self.h[t] - y[t]
            deltaOut = delta + deltaOut
            deltaState = deltaOut * o[t] * (1 - tanh(C[t]) ** 2) + deltaState[t + 1] * f[t + 1]
            deltaC = deltaState[t] * i[t] * (1 - C[t] ** 2)
            deltaI = deltaState[t] * C[t] * i[t] * (1 - i[t])
            deltaF = deltaState[t] * C[t - 1] * f[t] * (1 - f[t])
            deltaO = deltaOut[t] * tanh(C[t]) * o[t] * (1 - o[t])
            deltaGates[t] = np.array([deltaC[t], deltaI[t], deltaF[t], deltaO[t]])

        deltaW = 0
        deltaU = 0
        deltaB = 0
        for t in range(self.nInputs):
            deltaW += np.outer(deltaGates[t], self.x[t])
            deltaB += deltaGates[t]
            deltaU += np.outer(deltaGates[t + 1], self.h[t])
        return deltaW, deltaU, deltaB

    def SGD(self, X: list, y: list, batchSize: int, nEpochs: int, learningRate, lamb):
        """
        Implementation of Stochastic Gradient Descent

        It takes as input the network, the MNIST dataset, the MNIST labels of the dataset,
        the size of the batch to do gradient descent, the number of epochs it should run,
        the learning rate eta (I found the best eta to be in the order of 1s)
        and the regularization term lambda

        It returns a trained network
        """
        bestAcc = 0
        bestEpoch = 0
        eta = learningRate
        etaChangeEpoch = 0
        for epoch in range(nEpochs):
            batch = rd.sample(range(len(X)), batchSize)
            nablaW = np.zeros(self.nGates)
            nablaU = np.zeros(self.nGates)
            nablaB = np.zeros(self.nGates)
            for i in batch:
                self.setInput(X[i])
                # finding what should be modified based on this particular example
                deltaNablaW, deltaNablaU, deltaNablaB = LSTM.backProp(y[i])
                # passing this modifications to our overall modifications matrices
                nablaW += deltaNablaW
                nablaU += deltaNablaU
                nablaB += deltaNablaB

            # applying the changes to our network
            self.b = self.b - eta * (nablaB / batchSize)
            self.w = self.w - eta * (nablaW / batchSize) - eta * (lamb / batchSize) * self.w
            self.u = self.u - eta * (nablaU / batchSize) - eta * (lamb / batchSize) * self.u
            acc, outputs = testNetwork(X, y, nTests=batchSize)
            if acc > bestAcc:
                bestAcc = acc
                bestEpoch = epoch
            print(f'learningRate: {learningRate} epochs: {epoch} acc: {acc}, outputs: {outputs}')
        print(f'best acc: {bestAcc} on epoch: {bestEpoch}')

lstm = LSTM(nInputs=500, nFeatures=1, nCells=500, nOutputs=1, batchSize=1)
lstm.SGD(train_X, train_y, batchSize=100, nEpochs=100, learningRate = 1, lamb = 0)