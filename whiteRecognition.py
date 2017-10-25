from numpy import exp, array, random, dot

class NeuralNetwork():

    def __init__(self):
        self.synaptic_weights = [[0.3522301],[0.3522301],[0.3522301]]

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        output = True if self.__sigmoid(dot(inputs, self.synaptic_weights)) >= 0.70642878 else False
        return output