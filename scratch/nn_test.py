import numpy as np

from typing import Any, Callable
import random

from numpy.typing import NDArray


from nn import FeedforwardNN


def random_init() -> float:
    """
    Random weight initialization function.
    Returns a random float in the range [-1, 1].
    """
    return random.random() * 2 - 1


nn = FeedforwardNN([2, 3, 2], random_init)
nn.set_weights(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
nn.set_weights(
    np.array([0.7, -0.2, 1, 0, 0.1, -1, 0.2, -0.1, -0.3, 1, -0.6, -1, 0.3, 0, 0.3, 0.1, 0.2])
)
print(nn.get_weights())
print(nn.forward(np.array([1, -1])))


# nn = FeedforwardNN([4, 4], random_init)

# print(nn.weights)
# print(nn.biases)
# print(nn.forward(np.array([1, 0, 0, 1])))
# print(nn.get_weights())
# nn.set_weights(nn.get_weights())
# print(nn.get_weights())
# print(nn.forward(np.array([1, 0, 0, 1])))

# nn = FeedforwardNN([2, 3, 2], random_init)
# nn.set_weights(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
# nn.set_weights(
#     np.array([0.7, -0.2, 1, 0, 0.1, -1, 0.2, -0.1, -0.3, 1, -0.6, -1, 0.3, 0, 0.3, 0.1, 0.2])
# )
# print(nn.get_weights())
# print(nn.forward(np.array([1, -1])))
