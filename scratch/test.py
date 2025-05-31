def determine_weight_len(layers: list[int]) -> int:
    total_weights = 0
    for i in range(len(layers) - 1):
        total_weights += (layers[i] + 1) * layers[i + 1]  # +1 for bias
    return total_weights


# Example
layers = [9, 5, 3]
print(determine_weight_len(layers))  # Output: 32
