
import numpy as np
from typing import Any, Callable
import random

from numpy.typing import NDArray

class FeedforwardNN:
    def __init__(self, layers: list[int], weight_init_fun: Callable[[], float]):
        """
        layers: List of layer sizes, e.g., [3, 5, 2] for 3 input neurons, 5 hidden, 2 output
        weight_init_fun: A function returning a float, used to initialize weights
        """
        self.layers: list[int] = layers
        self.weights: list[NDArray[np.float64]] = []
        self.biases: list[NDArray[np.float64]] = []

        for i in range(len(layers) - 1):
            # weight matrix for layer i to layer i+1
            # shape: (layers[i+1], layers[i])
            # bias vector for layer i+1
            # shape: (layers[i+1],)
            weight_matrix = np.array([[weight_init_fun() for _ in range(layers[i])] for _ in range(layers[i+1])])
            bias_vector = np.array([weight_init_fun() for _ in range(layers[i+1])])

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    # TODO: activation function configuration
    def sigmoid(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x
        #return 1 / (1 + np.exp(-x))

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform a forward pass.
        x: Input vector (1D numpy array)
        Returns: Output of the final layer
        """
        activation = x
        for W, b in zip(self.weights, self.biases):
            z = W @ activation + b
            activation = self.sigmoid(z)
        return activation

    def get_weights(self) -> NDArray[np.float64]:
        """
        Returns the weights of the neural network as a numpy array.
        """
        
        # return weights and biases as a single numpy array
        # w_layer1 | bias_layer1 | w_layer2 | bias_layer2 | ...

        genes = []
        for W, b in zip(self.weights, self.biases):
            genes.append(W.flatten())
            genes.append(b.flatten())
        return np.concatenate(genes)

    # TOOO: add check for correct size
    def set_weights(self, weights: NDArray[np.float64]) -> None:
        """
        Set the weights of the neural network from a numpy array.
        weights: 1D numpy array containing the weights
        """
        idx = 0
        for i in range(len(self.layers) - 1):
            # get the number of weights for layer i
            num_weights = self.layers[i] * self.layers[i + 1]
            # get the number of biases for layer i+1
            num_biases = self.layers[i + 1]

            # reshape the weights and biases
            W = weights[idx:idx+num_weights].reshape((self.layers[i + 1], self.layers[i]))
            b = weights[idx+num_weights:idx+num_weights + num_biases]

            # set the weights and biases
            self.weights[i] = W
            self.biases[i] = b
            idx += num_weights + num_biases


def random_init() -> float:
    """
    Random weight initialization function.
    Returns a random float in the range [-1, 1].
    """
    return random.random() * 2 - 1

# nn = FeedforwardNN([4, 4], random_init)

# print(nn.weights)
# print(nn.biases)
# print(nn.forward(np.array([1, 0, 0, 1])))
# print(nn.get_weights())
# nn.set_weights(nn.get_weights())
# print(nn.get_weights())
# print(nn.forward(np.array([1, 0, 0, 1])))

nn = FeedforwardNN([2, 3, 2], random_init)
nn.set_weights(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
nn.set_weights(np.array([0.7, -0.2, 1, 0, 0.1, -1, 0.2, -0.1, -0.3, 1, -0.6, -1, 0.3, 0, 0.3, 0.1, 0.2]))
print(nn.get_weights())
print(nn.forward(np.array([1, -1])))