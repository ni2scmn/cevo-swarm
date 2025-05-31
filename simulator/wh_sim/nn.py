import random
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class FeedforwardNN:
    def __init__(
        self,
        layers: list[int],
        weight_init: Callable[[], float] | NDArray[np.float64],
        activation_fun: Callable[[NDArray[np.float64]], NDArray[np.float64]]
        | list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
        | None = None,
    ):
        """
        layers: List of layer sizes, e.g., [3, 5, 2] for 3 input neurons, 5 hidden, 2 output
        weight_init_fun: A function returning a float, used to initialize weights
        """
        self.layers: list[int] = layers
        self.weights: list[NDArray[np.float64]] = []
        self.biases: list[NDArray[np.float64]] = []

        self.activation_fun: list[Callable[[NDArray[np.float64]], NDArray[np.float64]]]
        # if activation_fun is None, use identity function
        # if activation_fun is a list, use that list
        # if activation_fun is a function, use that function for all layers
        if activation_fun is None:
            self.activation_fun = [lambda x: x] * (len(layers) - 1)
        elif isinstance(activation_fun, list):
            if len(activation_fun) != len(layers) - 1:
                raise ValueError("activation_fun must be a list of length len(layers) - 1")
            self.activation_fun = activation_fun
        else:
            self.activation_fun = [activation_fun] * (len(layers) - 1)

        # check if weight_init is a function
        if callable(weight_init):
            for i in range(len(layers) - 1):
                # weight matrix for layer i to layer i+1
                # shape: (layers[i+1], layers[i])
                # bias vector for layer i+1
                # shape: (layers[i+1],)
                weight_matrix = np.array(
                    [[weight_init() for _ in range(layers[i])] for _ in range(layers[i + 1])]
                )
                bias_vector = np.array([weight_init() for _ in range(layers[i + 1])])

                self.weights.append(weight_matrix)
                self.biases.append(bias_vector)
        else:
            if len(weight_init) != self._determine_weight_len():
                raise ValueError(
                    f"Expected {self._determine_weight_len()} weights, got {len(weight_init)}"
                )
            for i in range(len(layers) - 1):
                self.weights.append(np.array([]))
                self.biases.append(np.array([]))
            self.set_weights(weight_init)

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform a forward pass.
        x: Input vector (1D numpy array)
        Returns: Output of the final layer
        """
        activation = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ activation + b
            activation = self.activation_fun[i](z)
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

    def set_weights(self, weights: NDArray[np.float64]) -> None:
        """
        Set the weights of the neural network from a numpy array.
        weights: 1D numpy array containing the weights
        """
        if len(weights) != self._determine_weight_len():
            raise ValueError(f"Expected {self._determine_weight_len()} weights, got {len(weights)}")
        idx = 0
        for i in range(len(self.layers) - 1):
            # get the number of weights for layer i
            num_weights = self.layers[i] * self.layers[i + 1]
            # get the number of biases for layer i+1
            num_biases = self.layers[i + 1]

            # reshape the weights and biases
            W = weights[idx : idx + num_weights].reshape((self.layers[i + 1], self.layers[i]))
            b = weights[idx + num_weights : idx + num_weights + num_biases]

            # set the weights and biases
            self.weights[i] = W
            self.biases[i] = b
            idx += num_weights + num_biases

    def _determine_weight_len(self) -> int:
        total_weights = 0
        for i in range(len(self.layers) - 1):
            total_weights += (self.layers[i] + 1) * self.layers[i + 1]  # +1 for bias
        return total_weights


class NNBeliefSpace:
    def __init__(self, bs_nn_weights: NDArray[np.float64]):
        self.bs_nn_weights: NDArray[np.float64] = bs_nn_weights

    def get_weights(self) -> NDArray[np.float64]:
        return self.bs_nn_weights

    def set_weights(self, weights: NDArray[np.float64]) -> None:
        self.bs_nn_weights = weights


# Activation functions
########################################


def sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-x))


def tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)


def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0, x)


def leaky_relu(x: NDArray[np.float64], alpha: float = 0.01) -> NDArray[np.float64]:
    return np.where(x > 0, x, alpha * x)


def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# Weight initialization functions
########################################


def random_weight_init() -> float:
    return random.uniform(-1, 1)  # Random weight initialization in range [-1, 1]


def zero_weight_init() -> float:
    return 0.0  # Zero weight initialization
