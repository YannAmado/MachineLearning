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



class LSTM:
    """
    h = array of outputs
    x = array of inputs
    """

    def __init__(self, nInputs, nFeatures, nUnits, nOutputs, batchSize):
        nGates = 4
        if nUnits < nOutputs:
            print('the number of cells cannot be less than the number of outputs')

        # [t][i][j], t = timestep, i = which batch, j = which cell or feature
        x = np.zeros((nInputs, batchSize, nFeatures))
        h = np.zeros((nInputs, batchSize, nUnits))
        i = np.zeros((nInputs, batchSize, nUnits))
        f = np.zeros((nInputs, batchSize, nUnits))
        o = np.zeros((nInputs, batchSize, nUnits))
        tildeC = np.zeros((nInputs, batchSize, nUnits))
        C = np.zeros((nInputs, batchSize, nUnits))
        wxScale = 1 / np.sqrt(nFeatures * nUnits)
        whScale = 1 / np.sqrt(nUnits * nUnits)
        Wxi = np.random.normal(loc=0, scale=wxScale, size=[nFeatures, nUnits])
        Wxf = np.random.normal(loc=0, scale=wxScale, size=[nFeatures, nUnits])
        Wxc = np.random.normal(loc=0, scale=wxScale, size=[nFeatures, nUnits])
        Wxo = np.random.normal(loc=0, scale=wxScale, size=[nFeatures, nUnits])
        Whi = np.random.normal(loc=0, scale=whScale, size=[nUnits, nUnits])
        Whf = np.random.normal(loc=0, scale=whScale, size=[nUnits, nUnits])
        Whc = np.random.normal(loc=0, scale=whScale, size=[nUnits, nUnits])
        Who = np.random.normal(loc=0, scale=whScale, size=[nUnits, nUnits])
        bi = np.random.normal(loc=0, scale=1, size=[nUnits])
        bf = np.random.normal(loc=0, scale=1, size=[nUnits])
        bc = np.random.normal(loc=0, scale=1, size=[nUnits])
        bo = np.random.normal(loc=0, scale=1, size=[nUnits])
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
        self.Wx = np.array([self.Wxc, self.Wxi, self.Wxf, self.Wxo])
        self.Wh = np.array([self.Whc, self.Whi, self.Whf, self.Who])
        self.bi = bi
        self.bf = bf
        self.bc = bc
        self.bo = bo
        self.b = np.array([self.bc, self.bi, self.bf, self.bo])
        self.nUnits = nUnits
        self.nGates = nGates
        self.nInputs = nInputs
        self.batchSize = batchSize

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
        for t in range(1, self.nUnits):
            self.x[t] = x[t]
            self.C[t], self.h[t] = self.getNewState(t, x[t])
        return

    def testNetwork(self, test_X, test_y, nTests: int):
        """
        A function to test our network

        It returns the overall accuracy and the numbers our network guessed
        """

        correctOutput = 0
        X = test_X[:nTests]
        y = test_y[:nTests]
        outputs = np.zeros(10)
        for i in range(nTests):
            self.setInput(X[i])
            networkOutput = self.h[-1]
            outputs[networkOutput] += 1
            # print(f"number: {y[i]}, networkOutput: {networkOutput}, activations: {net.a[-1]}")
            if y[i] == networkOutput:
                correctOutput += 1
        acc = correctOutput / nTests
        return acc, outputs

    def backProp(self, y):
        i = self.i
        f = self.f
        C = self.C
        tildeC = self.tildeC
        o = self.o
        delta = self.h[-1] - y
        Wx = self.Wx
        Wh = self.Wh


        deltaH = np.zeros_like(self.h)
        triangle = np.zeros(self.nInputs)
        triangleH = np.zeros_like(self.h)
        deltaC = np.zeros_like(C)
        deltaTildeC = np.zeros_like(tildeC)
        deltaI = np.zeros_like(i)
        deltaF = np.zeros_like(f)
        deltaO = np.zeros_like(o)
        deltaX = np.zeros_like(self.x)
        deltaGates = np.zeros((self.nInputs, self.nGates, np.shape(self.C[0])[0], np.shape(self.C[0])[1]))

        # appending one extra element because of the first interaction in the if below
        # deltaState and f have t + 1
        deltaC = np.append(deltaC, np.zeros((1,self.batchSize, self.nUnits)), axis=0)
        f      = np.append(f, np.zeros((1,self.batchSize, self.nUnits)), axis=0)
        deltaGates = np.append(deltaGates, np.zeros((1, self.nGates, np.shape(self.C[0])[0], np.shape(self.C[0])[1])), axis=0)
        for t in range(self.nInputs - 1, 0, -1):
            # not too sure about delta because I only have 1 output
            #delta = self.h[t] - y[t]
            triangle[t] = self.h[t] - y
            # I need the dot product to have dimensions batchsize, 1
            triangleH[t] = np.dot(np.transpose(Wh), deltaGates[t + 1])
            deltaH[t] = triangle[t] + triangleH[t]
            deltaC[t] = deltaH[t] * o[t] * (1 - pow(tanh(C[t]), 2)) + deltaC[t + 1] * f[t + 1]
            deltaTildeC[t] = deltaC[t] * i[t] * (1 - pow(tildeC[t], 2))
            deltaI[t] = deltaC[t] * tildeC[t] * i[t] * (1 - i[t])
            deltaF[t] = deltaC[t] * C[t - 1] * f[t] * (1 - f[t])
            deltaO[t] = deltaH[t] * tanh(C[t]) * o[t] * (1 - o[t])
            deltaX[t] = np.dot(np.transpose(Wx), deltaGates[t + 1])
            triangleH[t - 1] = np.dot(np.transpose(Wh), deltaGates[t + 1])
        deltaWx = np.zeros_like(self.Wx)
        deltaWh = np.zeros_like(self.Wh)
        deltaB = np.zeros_like(self.b)
        for t in range(self.nInputs):
            deltaWx += np.outer(deltaGates[t], self.x[t])
            deltaWh += np.outer(deltaGates[t + 1], self.h[t])
            deltaB += deltaGates[t + 1]
        return deltaWx, deltaWh, deltaB

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
            nablaWx = np.zeros_like(self.Wx)
            nablaWh = np.zeros_like(self.Wh)
            nablaB = np.zeros_like(self.b)
            for i in batch:
                self.setInput(X[i])
                # finding what should be modified based on this particular example
                deltaNablaWx, deltaNablaWh, deltaNablaB = self.backProp(y[i])
                # passing this modifications to our overall modifications matrices
                nablaWx += deltaNablaWx
                nablaWh += deltaNablaWh
                nablaB += deltaNablaB

            # applying the changes to our network
            self.b = self.b - eta * (nablaB / batchSize)
            self.Wx = self.Wx - eta * (nablaWx / batchSize) - eta * (lamb / batchSize) * self.Wx
            self.Wh = self.Wh - eta * (nablaWh / batchSize) - eta * (lamb / batchSize) * self.Wh
            acc, outputs = self.testNetwork(X, y, nTests=batchSize)
            if acc > bestAcc:
                bestAcc = acc
                bestEpoch = epoch
            print(f'learningRate: {learningRate} epochs: {epoch} acc: {acc}, outputs: {outputs}')
        print(f'best acc: {bestAcc} on epoch: {bestEpoch}')

lstm = LSTM(nInputs=500, nFeatures=1, nUnits=1, nOutputs=1, batchSize=1)
lstm.SGD(train_X, train_y, batchSize=100, nEpochs=100, learningRate = 1, lamb = 0)