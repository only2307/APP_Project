import numpy as np
from layer import Layer 
from correlate import *

class CNNLayer(Layer): #CNN su dung ham correlate o tren
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        np.random.seed(0)
        self.kernels = np.random.randn(*self.kernels_shape)
        np.random.seed(0)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += correlate(self.input[j], self.kernels[i, j], 1, "valid")
                #self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        kernels_rot_180=np.rot90(self.kernels,2)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = correlate(self.input[j], output_gradient[i], 1, "valid")
                input_gradient[j] += correlate(output_gradient[i], kernels_rot_180[i, j], 1, "full")
                #kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                #input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient