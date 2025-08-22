# Programs desc

**neuralnet.py** is the main program, where MLP is built from scratch. <br/>
**nonlins_test.py** is created to test various non linear structures using neural network.

**For sigmoid -**
activation function - 1/(1+exp(-z))     <br/>
activation derivative - a * (1.0 - a)
loss function -> -y*log(yhat)-(1-y)log(1-yhat)


**For softmax**
activation function - exp(z - max(z)) / sum(z, axis=-1)     _Sum across row in columns_
activation derivative - (yhat - y)*x
loss function -> -sum(y[k] * log(yhat[k])) *k=0-n_classes*

**Usage**
use if else in activation, derivation and loss functions.<br/>

-> If binary classification problem                      <br/>
    -> Use sigmoid in all places (while adding layers)     <br/>
-> If multi-class classification problem                <br/>
    -> Use sigmoid in input and all hidden layers     <br/>
    -> Use softmax in output layer                      <br/>
    -> Output layer n_neurons = n_classes                <br/>
    -> Pass one hot encoded ytrain in fit method



# Variables

self.cost - All losses in every epochs    <br/>
self.layerWeights - Assigned and updated weights on every layer    <br/>
self.layerBiases - Assigned and updated biases on every layer    <br/>
self.layerOutputs - Outputs of every neuron on every layer. (Added, updated and used during backpropagation)    <br/>
self.layerDeltas - All deltas of all layers. (Added, updated and used during backpropagation)    <br/>
self.activationChoices - Activation functions assigned for each layer    <br/>
self.loss_function - String that specifies which loss function to use    <br/>



# XOR problem
Solving limitations of Perceptron using neural network.


**Architechture - 2(inputs) - 4 - 2 - 1(output)**

With lr 0.05, epochs 1000,<br/>
    Accuracy 98.67%<br/>
    loss - 0.7 -> 0.0476

With lr 0.1, epochs 1000,<br/>
    Accuracy 99.33%<br/>
    loss - 0.7 -> 0.0476


**Architechture - 2(inputs) - 2 - 1(output)**

With lr 0.1, epochs 1000,<br/>
    Accuracy 85.33% _(Boundary is linear)_    <br/>
    loss - 0.7 -> 0.3

With lr 0.1, epochs 5000,<br/>
    Accuracy 83.33% _(Boundary is linear)_<br/>
    loss - 0.7 -> 0.3

With lr 0.5, epochs 5000,<br/>
    Accuracy 82.67% _(Boundary is linear, after 1000 epochs no change in loss(0.3) )_<br/>
    loss - 0.7 -> 0.3


**Architechture - 2(inputs) - 2 - 2 - 1(output)**

With lr 0.1, epochs 1000,<br/>
    Accuracy 82% _(Boundary is linear)_    <br/>
    loss - 0.7 -> 0.3



# Spiral dataset

**Architechture - 2(inputs) - 16-16-16-16-4 - 1(output)**

With lr 0.01, epochs 10,000,<br/>
    _8000 is enough_    <br/>
    Accuracy 97 - 100%
    loss - 0.7 -> 0.018

**Architecture - 2 - ( 25 * 5 ) - 5 - 1**
With lr 0.1, epochs 10,000,<br/>
    _2000 is enough, reached 0.002 loss. On 4000 epochs reached 0.0017 loss_    <br/>
    Accuracy 100%
    loss - 0.7 -> 0.00100113

