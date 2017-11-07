import cv2
import numpy as np
import itertools
from math import hypot
from scipy import stats

from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((11, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


class Classifier:
    def __init__(self, image):
        self.__im = image

    @staticmethod
    def dist(p, q):
        return hypot(p[1] - p[0], q[1] - q[0])

    def classify(self):
        gray = cv2.cvtColor(self.__im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        width, height = thresh.shape
        #status = np.ones((width, height), dtype=bool)

        upper, lower = max([width, height]), min([width, height])
        index = 0 if lower == width else 1
        nums = [i for i in range(upper)]
        data, dists = [], []

        for value in itertools.product(nums, repeat=2):
            if value[index] < lower and thresh[value[0], value[1]] == 0:
                data.append(value)

        lim = len(data)
        for i in range(lim):
            for j in range(i + 1, lim):
                dists.append(Classifier.dist(data[i], data[j]))

        if len(dists) == 0:
            return None

        dists = np.array(dists)
        mean = np.mean(dists)
        err_mean = stats.sem(dists, ddof=1)
        median = np.median(dists)
        mode = stats.mode(dists)[0]
        var = np.var(dists, ddof=1)
        st_dev = np.std(dists, ddof=1)
        skew = stats.skew(dists)
        kurt = stats.kurtosis(dists)

        """"
        print("---------------------------")
        print("N: {}".format(lim))
        print("width: {}".format(width))
        print("height: {}".format(height))
        print("mean: {}".format(mean))
        print("error mean: {}".format(err_mean))
        print("median: {}".format(median))
        print("mode: {}".format(mode))
        print("var: {}".format(var))
        print("st_dev: {}".format(st_dev))
        print("skew: {}".format(skew))
        print("kurt: {}".format(kurt))
        """

        r = [lim, width, height, mean, err_mean, median, mode[0], var, st_dev, skew, kurt]

        return r
