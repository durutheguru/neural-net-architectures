
from neuron import *
from sigmoid import *
from mse import *

import numpy as np


class Network:

    def __init__(self):
        self.h1 = Neuron()
        self.h2 = Neuron()
        self.o1 = Neuron()


    def feed_forward(self, x):
        out_h1 = self.h1.feed_forward(x)
        out_h2 = self.h2.feed_forward(x)
        out_o1 = self.o1.feed_forward(np.array([out_h1, out_h2]))
        return out_o1
    

    def train(self, data, y_actuals):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, y_actuals):
                self.h1.feed_forward(x)
                self.h2.feed_forward(x)
                self.o1.feed_forward(np.array([self.h1.output, self.h2.output]))

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - self.o1.output)

                # Neuron o1
                d_ypred_d_w5 = self.h1.output * deriv_sigmoid(self.o1.computed_sum)
                d_ypred_d_w6 = self.h2.output * deriv_sigmoid(self.o1.computed_sum)
                d_ypred_d_b3 = deriv_sigmoid(self.o1.computed_sum)

                d_ypred_d_h1 = self.o1.weights[0] * deriv_sigmoid(self.o1.computed_sum)
                d_ypred_d_h2 = self.o1.weights[1] * deriv_sigmoid(self.o1.computed_sum)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(self.h1.computed_sum)
                d_h1_d_w2 = x[1] * deriv_sigmoid(self.h1.computed_sum)
                d_h1_d_b1 = deriv_sigmoid(self.h1.computed_sum)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(self.h2.computed_sum)
                d_h2_d_w4 = x[1] * deriv_sigmoid(self.h2.computed_sum)
                d_h2_d_b2 = deriv_sigmoid(self.h2.computed_sum)

                # --- Update weights and biases
                # Neuron h1
                self.h1.weights[0] = self.h1.weights[0] - (learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1)
                self.h1.weights[1] = self.h1.weights[1] - (learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2)
                self.h1.bias = self.h1.bias - (learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1)

                # Neuron h2
                self.h2.weights[0] = self.h2.weights[0] - (learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3)
                self.h2.weights[1] = self.h2.weights[1] - (learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4)
                self.h2.bias = self.h2.bias - (learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2)

                # Neuron o1
                self.o1.weights[0] = self.o1.weights[0] - (learn_rate * d_L_d_ypred * d_ypred_d_w5)
                self.o1.weights[1] = self.o1.weights[1] - (learn_rate * d_L_d_ypred * d_ypred_d_w6)
                self.o1.bias = self.o1.bias - (learn_rate * d_L_d_ypred * d_ypred_d_b3)

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feed_forward, 1, data)
                loss = mse_loss(y_actuals, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))





data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

y_actuals = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = Network()
network.train(data, y_actuals)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feed_forward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feed_forward(frank)) # 0.039 - M

