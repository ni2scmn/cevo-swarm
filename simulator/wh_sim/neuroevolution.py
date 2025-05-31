import numpy as np

from typing import Any, Callable
import random

from numpy.typing import NDArray


def point_mutate(
    weights: NDArray[np.float64],
    mutation_rate: float,
    mutation: Callable[[], float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    mutation_mask = np.random.rand(*weights.shape) < mutation_rate

    mutation_values: NDArray[np.float64]
    if isinstance(mutation, np.ndarray):
        mutation_values = mutation
    elif callable(mutation):
        mutation_values = np.array(np.vectorize(mutation)(np.zeros_like(weights)), dtype=np.float64)

    weights = weights + mutation_mask * mutation_values
    return weights


def one_point_crossover(
    parent1: NDArray[np.float64], parent2: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    assert parent1.shape == parent2.shape
    flat_size = parent1.size
    point = np.random.randint(1, flat_size - 1)

    p1_flat, p2_flat = parent1.flatten(), parent2.flatten()
    child1 = np.concatenate([p1_flat[:point], p2_flat[point:]])
    child2 = np.concatenate([p2_flat[:point], p1_flat[point:]])

    return child1.reshape(parent1.shape), child2.reshape(parent2.shape)


def two_point_crossover(
    parent1: NDArray[np.float64], parent2: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    assert parent1.shape == parent2.shape
    assert parent1.shape == parent2.shape
    flat_size = parent1.size

    point1, point2 = sorted(np.random.choice(range(1, flat_size), size=2, replace=False))

    p1_flat, p2_flat = parent1.flatten(), parent2.flatten()

    child1 = np.concatenate([p1_flat[:point1], p2_flat[point1:point2], p1_flat[point2:]])

    child2 = np.concatenate([p2_flat[:point1], p1_flat[point1:point2], p2_flat[point2:]])

    return child1.reshape(parent1.shape), child2.reshape(parent2.shape)


def n_point_crossover(
    parent1: NDArray[np.float64], parent2: NDArray[np.float64], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    assert parent1.shape == parent2.shape
    flat_size = parent1.size
    assert n > 0 and n < flat_size, "n must be between 1 and number of genes - 1"

    crossover_points = np.sort(np.random.choice(range(1, flat_size), size=n, replace=False))
    crossover_points = np.concatenate([[0], crossover_points, [flat_size]])

    p1_flat, p2_flat = parent1.flatten(), parent2.flatten()
    child1, child2 = np.empty_like(p1_flat), np.empty_like(p2_flat)

    # Alternate segments between parents
    for i in range(len(crossover_points) - 1):
        start, end = crossover_points[i], crossover_points[i + 1]
        if i % 2 == 0:
            child1[start:end] = p1_flat[start:end]
            child2[start:end] = p2_flat[start:end]
        else:
            child1[start:end] = p2_flat[start:end]
            child2[start:end] = p1_flat[start:end]

    return child1.reshape(parent1.shape), child2.reshape(parent2.shape)
