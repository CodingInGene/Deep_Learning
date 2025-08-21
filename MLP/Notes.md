# Programs desc

**neuralnet.py** is the main program, where MLP is built from scratch
**nonlins_test.py** is created to test various non linear structures using neural network.



# Variables

self.cost - All losses in every epochs
self.layerWeights - Assigned and updated weights on every layer
self.layerBiases - Assigned and updated biases on every layer
self.layerOutputs - Outputs of every neuron on every layer. (Added, updated and used during backpropagation)
self.layerDeltas - All deltas of all layers. (Added, updated and used during backpropagation)
self.activationChoices - Activation functions assigned for each layer
self.loss_function - String that specifies which loss function to use



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
    Accuracy 85.33% _(Boundary is linear)_
    loss - 0.7 -> 0.3

With lr 0.1, epochs 5000,<br/>
    Accuracy 83.33% _(Boundary is linear)_<br/>
    loss - 0.7 -> 0.3

With lr 0.5, epochs 5000,<br/>
    Accuracy 82.67% _(Boundary is linear, after 1000 epochs no change in loss(0.3) )_<br/>
    loss - 0.7 -> 0.3


**Architechture - 2(inputs) - 2 - 2 - 1(output)**

With lr 0.1, epochs 1000,<br/>
    Accuracy 82% _(Boundary is linear)_
    loss - 0.7 -> 0.3



# Spiral dataase

**Architechture - 2(inputs) - 16-16-16-16-4 - 1(output)**

With lr 0.01, epochs 10,000,<br/>
    _8000 is enough_
    Accuracy 97 - 100%
    loss - 0.7 -> 0.018
