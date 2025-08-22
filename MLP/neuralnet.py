#Make a neural net
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs, make_classification, make_circles
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import random
import math
from collections import defaultdict

#Function block
def loadData(dataset):
    if dataset=="iris":
        iris = load_iris()      #Dictionary
        
        data = iris["data"]
        target = iris["target"]

        y=[]
        for i in target:
            if i==1:    #setosa
                y.append(1)
            else:
                y.append(0)
        return data[:,0 :2], np.array(target)
    
    if dataset=="xor":
        x = np.random.uniform(low=-1, high=1, size=(500,2))
        y = np.bitwise_xor(np.sign(x[:,0]).astype(int),np.sign(x[:,1]).astype(int))

        y = np.array([0 if i<0 else 1 for i in y])

        return x, y

class MultiLayerPerceptron:
    def __init__(self):
        self.lr = None
        self.epochs = None
        self.layerWeights = defaultdict(list)
        self.layerBiases = defaultdict(list)
        self.layerOutputs = defaultdict(list)
        self.layerDeltas = defaultdict(list)
        self.activationChoices = defaultdict(list)
        self.loss_function = None
        self.cost = []
    def add(self, n_nodes, inputs_dim=None, activation="sigmoid", loss_function=None):
        if inputs_dim != None:  #For input layer
            self.layerWeights[len(self.layerWeights)] = np.random.randn(inputs_dim, n_nodes) #np.zeros((inputs_dim, n_nodes))
        else:
            #Hidden layers
            prev_layer = len(self.layerWeights[len(self.layerWeights)-1][0])   #Column display how many nodes are present

            self.layerWeights[len(self.layerWeights)] = np.random.randn(prev_layer, n_nodes) #np.zeros((prev_layer, n_nodes))

        self.activationChoices[len(self.activationChoices)] = activation    #Store assigned activation names
        self.layerBiases[len(self.layerBiases)] = np.zeros((n_nodes))       #Store biases
        self.loss_function = loss_function

    def activation(self,z, activation):
        if activation=="sigmoid":
            return 1/(1+np.exp(-z))     #Sigmoid activation
        if activation=="softmax":
            exp_z = np.exp(z - np.max(z))  # numerical stability
            return exp_z / np.sum(exp_z, axis=-1)

    def findLoss(self, y_hat, yactual):  #Expects single row of y
        if self.loss_function == "binary_crossentropy":
            loss = -(yactual*np.log(y_hat)) - ((1-yactual)*np.log(1-y_hat))
        if self.loss_function == "categorical_crossentropy":
            loss = -np.sum(yactual * np.log(y_hat))
        return loss

    def activation_derivative(self, y_hat, yactual=None, row=None, activation=None):   #yactual, row are only used on softmax function
        if activation == "sigmoid":
            return y_hat * (1.0 - y_hat)
        if activation == "softmax":
            return np.dot((y_hat - yactual),row)    #For softmax 

    def forwardPass(self,row):    #Takes batch of inputs
        output = row
        for i in range(len(self.layerWeights)):
            z = np.dot(output, self.layerWeights[i]) + self.layerBiases[i]
            output = self.activation(z, activation=self.activationChoices[i])
            self.layerOutputs[i] = output   #Storing each layer's output
        return output   #Send last layers output (y_hat)

    def backpropagation(self, y_hat, yactual, row):   #yactual, row are only used on softmax function
        #Errors
        for i in reversed(range(len(self.layerWeights))):
            if i == len(self.layerWeights)-1:   #Output layer
                if self.activationChoices[i] == "sigmoid":
                    self.layerDeltas[i] = (y_hat - yactual) * self.activation_derivative(y_hat, activation=self.activationChoices[i])
                elif self.activationChoices[i] == "softmax":
                    self.layerDeltas[i] = (y_hat - yactual) #No need to use derivation here, as it is (yhat-y)
            else:
                da = self.activation_derivative(self.layerOutputs[i], yactual, row, activation=self.activationChoices[i])   #Derivation of activation
                self.layerDeltas[i] = np.dot(self.layerWeights[i+1], self.layerDeltas[i+1]) * da

        #Weight and bias update
        for i in reversed(range(len(self.layerWeights))):
            if i==0:
                a_prev = row    #When in 1st layer pass input data
            else:
                a_prev = self.layerOutputs[i-1]

            dW = np.outer(a_prev, self.layerDeltas[i])  #np.outer to match delta and input neurons
            self.layerWeights[i] -= self.lr * dW
            self.layerBiases[i] -= self.lr * self.layerDeltas[i]

    def fit(self, xtrain, ytrain, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        for i in range(epochs):
            loss = 0
            for row in range(xtrain.shape[0]):
                #Forward pass
                y_hat = self.forwardPass(xtrain[row])
                loss += self.findLoss(y_hat, ytrain[row])    #Calculating loss from last layers output
                #Backward pass
                self.backpropagation(y_hat, ytrain[row], xtrain[row])
            loss =  loss / xtrain.shape[0]   #Average loss
            print(f"Epoch - {i}, Loss - {loss}")
            self.cost.append(loss)  #Keeping track of loss
        self.cost = np.array(self.cost)

    def predict(self, xtest):
        y_pred = []
        for i in xtest:
            y_hat = self.forwardPass(i)
            y_pred.append(y_hat)
        return np.array(y_pred)




#Main block
if __name__=="__main__":
    X, y = loadData(dataset="iris")

    #Split
    xtrain, xtest, ytrain, ytest = train_test_split(X,y, train_size=0.7)
    print(xtrain.shape, ytest.shape)

    model = MultiLayerPerceptron()
    model.add(25, inputs_dim=2, activation="sigmoid")
    model.add(25, activation="sigmoid")
    model.add(3, activation="softmax", loss_function="categorical_crossentropy")

    lr = 0.1
    epochs = 1000

    #print(ytrain.reshape(-1,1))

    onehot = OneHotEncoder(sparse_output=False)
    ytrain_new = onehot.fit_transform(ytrain.reshape(-1,1))
    ytest_new = onehot.fit_transform(ytest.reshape(-1,1))


    model.fit(xtrain, ytrain_new, learning_rate=lr, epochs=epochs)

    print(ytrain_new[:10])

    # print(model.layerWeights)
    # print(model.layerBiases)

    y_pred = model.predict(xtest)
    print(y_pred)
    #y_pred = np.where(y_pred>=0.5, 1, 0)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)

    #Accuracy
    acc = accuracy_score(ytest, y_pred)
    print("\nAcccuracy",round(acc,4))

    #Plot
    fig, (ax1,ax2) = plt.subplots(nrows=2)
    ax1.plot(np.arange(model.cost.shape[0]), model.cost)
    ax2.scatter(xtrain[:,0], xtrain[:,1], c=ytrain)


    # Create mesh grid for decision boundary
    x_min, x_max = xtrain[:, 0].min() - 1, xtrain[:, 0].max() + 1
    y_min, y_max = xtrain[:, 1].min() - 1, xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict over mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax2.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.2)
    ax2.contour(xx, yy, Z, colors='k', levels=[0, 1, 2], 
                linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    plt.suptitle(f"Accuracy {round(acc,4)*100}%\nlr {lr}, Epochs {epochs}")
    plt.show()
