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
np.random.seed(42)
rd.seed(42)



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


        #remove this
        self.Wxc = np.array([0.45, 0.25])
        self.Wxi = np.array([0.95, 0.8])
        self.Wxf = np.array([0.7, 0.45])
        self.Wxo = np.array([0.6, 0.4])
        self.Whi = np.array([0.8])
        self.Whf = np.array([0.1])
        self.Who = np.array([0.25])
        self.Whc = np.array([0.15])


        self.Wx = np.array([self.Wxc, self.Wxi, self.Wxf, self.Wxo])
        self.Wh = np.array([self.Whc, self.Whi, self.Whf, self.Who])

        self.bi = bi
        self.bf = bf
        self.bc = bc
        self.bo = bo

        #remove this
        self.bc = np.array([0.2])
        self.bi = np.array([0.65])
        self.bf = np.array([0.15])
        self.bo = np.array([0.1])
        self.b = np.array([self.bc, self.bi, self.bf, self.bo])
        self.nUnits = nUnits
        self.nGates = nGates
        self.nInputs = nInputs
        self.batchSize = batchSize
        self.nFeatures = nFeatures



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

    def outputGate(self, t):
        if t == 0:
            self.o[t] = sigmoid(np.dot(self.x[t], self.Wxo) + self.bo)
        else:
            self.o[t] = sigmoid(np.dot(self.h[t - 1], self.Who) + np.dot(self.x[t], self.Wxo) + self.bo)
        self.h[t] = self.o[t] * tanh(self.C[t])
        return self.h[t]

    def getNewState(self, t):
        self.forgetGate(t)
        self.inputGate(t)
        self.C[t] = self.C[t - 1] * self.f[t] + self.i[t] * self.tildeC[t]
        self.h[t] = self.outputGate(t)
        return self.C[t], self.h[t]

    def setInput(self, x):
        self.x[0] = x[0]
        self.getNewState(0)
        for t in range(1, self.nInputs):
            self.x[t] = x[t]
            self.getNewState(t)
        return

    def testNetwork(self, test_X, test_y, nTests: int):
        """
        A function to test our network

        It returns the overall accuracy and the numbers our network guessed
        """

        X = test_X[:nTests]
        y = test_y[:nTests]
        outputs = np.zeros(2)
        mse = 0
        for i in range(0, nTests, self.batchSize):
            # shaping the input to have the same dimensions as x[t] and h[t]
            batchX = np.zeros((self.nInputs, self.batchSize, self.nFeatures))
            for j in range(self.batchSize):
                batchX[:, j] = np.reshape(X[i + j], (-1, self.nFeatures))
            batchY = np.reshape(y[i:i + self.batchSize], (-1, self.nUnits))
            self.setInput(batchX)
            networkOutputs = np.where(self.h[-1] < 0.5, 0, 1)
            for out in networkOutputs:
                outputs[out] += 1
            # print(f"number: {y[i]}, networkOutput: {networkOutput}, activations: {net.a[-1]}")
            for j in range(self.batchSize):
                mse += pow((batchY[j] - networkOutputs[j]), 2)
        mse = mse/self.batchSize
        return mse, outputs

    def backProp(self, y):
        i = self.i
        f = self.f
        C = self.C
        tildeC = self.tildeC
        o = self.o
        Wx = self.Wx
        Wh = self.Wh


        deltaH = np.zeros_like(self.h)
        triangle = np.zeros((self.nInputs, self.batchSize, self.nUnits))
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

        # C has t - 1 in deltaF
        C = np.append(C, np.zeros((1,self.batchSize, self.nUnits)), axis=0)

        # not too sure about delta because I only have 1 output
        for t in range(self.nInputs - 1, -1, -1):
            # I need the dot product to have dimensions batchsize, 1
            # not sure if its working, have to see after t is bigger than 0
            triangle[t] = self.h[t] - y[t]
            """for j in range(self.batchSize):
                triangleH[t, j] = np.dot(np.transpose(Wh), deltaGates[t + 1,:,j])"""
            deltaH[t] = triangle[t] + triangleH[t]
            deltaC[t] = deltaH[t] * o[t] * (1 - pow(tanh(C[t]), 2)) + deltaC[t + 1] * f[t + 1]
            deltaTildeC[t] = deltaC[t] * i[t] * (1 - pow(tildeC[t], 2))
            deltaI[t] = deltaC[t] * tildeC[t] * i[t] * (1 - i[t])
            deltaF[t] = deltaC[t] * C[t - 1] * f[t] * (1 - f[t])
            deltaO[t] = deltaH[t] * tanh(C[t]) * o[t] * (1 - o[t])
            deltaGates[t] = np.array([deltaTildeC[t], deltaI[t], deltaF[t], deltaO[t]])
            for j in range(self.batchSize):
                deltaX[t, j] = np.dot(np.transpose(Wx), deltaGates[t, :, j])[:, 0]
                # even though eventually it will calculate triangleH[-1], it will not be used
                triangleH[t - 1, j] = np.dot(np.transpose(Wh), deltaGates[t, :, j])[:, 0]
        deltaWx = np.zeros_like(self.Wx)
        deltaWh = np.zeros_like(self.Wh)
        deltaB = np.zeros_like(self.b)
        for t in range(self.nInputs):
            for j in range(self.batchSize):
                for k in range(self.nUnits):
                    deltaWx += np.outer(deltaGates[t, :, j, k], self.x[t, j])
                    if t < self.nInputs - 1: # deltaWh is until T - 1
                        deltaWh += np.outer(deltaGates[t + 1, :, j, k], self.h[t, j])
                    #deltaWx[:, :, k] += np.outer(deltaGates[t, :, j, k], self.x[t, j])
                    #deltaWh[:, :, k] += np.outer(deltaGates[t + 1, :, j, k], self.h[t, j])
                # deltaB is from zero to T, different from many posts
                deltaB += deltaGates[t, :, j]
        return deltaWx, deltaWh, deltaB

    def SGD(self, X: list, y: list, SGDbatchSize: int, nEpochs: int, learningRate, lamb):
        """
        Implementation of Stochastic Gradient Descent

        It takes as input the network, the MNIST dataset, the MNIST labels of the dataset,
        the size of the batch to do gradient descent, the number of epochs it should run,
        the learning rate eta (I found the best eta to be in the order of 1s)
        and the regularization term lambda

        It returns a trained network
        """
        bestMse = 0
        bestEpoch = 0
        eta = learningRate
        etaChangeEpoch = 0
        for epoch in range(nEpochs):
            #batch = rd.sample(np.shape(X)[0], SGDbatchSize)
            nablaWx = np.zeros_like(self.Wx)
            nablaWh = np.zeros_like(self.Wh)
            nablaB = np.zeros_like(self.b)
            for i in range(0, SGDbatchSize, self.batchSize):
                # shaping the input to have the same dimensions as x[t] and h[t]
                """batchX = np.zeros((self.nInputs, self.batchSize, self.nFeatures))
                for j in range(self.batchSize):
                    batchX[:, j] = np.reshape(X[batch[i]], (-1, self.nFeatures))
                for j in range(self.nInputs):
                    batchX[i+j] = X[batch[i]]"""

                # shaping the output to have the same dimensions as h[t]
                #batchY = np.reshape(y[i:i + self.batchSize], (-1, self.nUnits))


                self.setInput(X)
                # finding what should be modified based on this particular example
                deltaNablaWx, deltaNablaWh, deltaNablaB = self.backProp(y)
                # passing this modifications to our overall modifications matrices
                nablaWx += deltaNablaWx
                nablaWh += deltaNablaWh
                nablaB += deltaNablaB

            # applying the changes to our network
            self.b = self.b - eta * (nablaB / SGDbatchSize)
            self.Wx = self.Wx - eta * (nablaWx / SGDbatchSize) - eta * (lamb / SGDbatchSize) * self.Wx
            self.Wh = self.Wh - eta * (nablaWh / SGDbatchSize) - eta * (lamb / SGDbatchSize) * self.Wh
            mse, outputs = self.testNetwork(X, y, nTests=SGDbatchSize)
            if mse > bestMse:
                bestMse = mse
                bestEpoch = epoch
            if epoch%100 == 0:
                print(f'learningRate: {learningRate} epochs: {epoch} mse: {mse}, outputs: {outputs}')
        print(f'best acc: {bestMse} on epoch: {bestEpoch}')


etas = [0.1, 1, 10]
for l in etas:
    print(f"learnign rate: {l}")
    train_X = np.array(([1, 2], [0.5, 3]))
    train_y = np.array((0.5,1.25))
    lstm = LSTM(nInputs=2, nFeatures=2, nUnits=1, nOutputs=1, batchSize=1)
    lstm.SGD(train_X, train_y, SGDbatchSize=1, nEpochs=200, learningRate=l, lamb=0)