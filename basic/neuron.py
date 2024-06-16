
import numpy as np
import sigmoid as si


class Neuron:
    def __init__(self):
        self.weights = np.array([np.random.normal(), np.random.normal()])
        self.bias = np.random.normal()

    def feed_forward(self, inputs):
        self.computed_sum = np.dot(self.weights, inputs) + self.bias
        self.output = si.sigmoid(self.computed_sum)
        return self.output


# weights = np.array([0, 1])
# bias = 4
# neuron = Neuron(weights, bias)

# x = np.array([2, 3])
# print(neuron.feed_forward(x))

