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
        self.weights: NDArray[np.float64],

        for i in range(len(layers) - 1):
            weight_matrix = np.array([[weight_init_fun() for _ in range(layers[i])] for _ in range(layers[i+1])])
            bias_vector = np.array([weight_init_fun() for _ in range(layers[i+1])])

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (1 + np.exp(-x))

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


def random_init() -> float:
    """
    
    """
    return random.random() * 2 - 1
    

nn = FeedforwardNN([2, 2], random_init)

print(nn.weights)
print(nn.biases)
print(nn.forward(np.array([1, 0])))