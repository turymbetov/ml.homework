import numpy as np
import random

edge_weight = 1
self_weight = -1
noise_power = -16

def noise(eps):
    return random.randint(0,9) * pow(10, eps)

def edges_to_adj_matrix(data: list[str], border) -> np.ndarray:
    adj_matrix = np.zeros((border, border))

    for i in range(border):
        adj_matrix[i][i] = self_weight

    for line in data:
        x, y = line.strip().split('\t')
        x = int(x)
        y = int(y)
        if x > border - 1:
            break

        if y <= border - 1:
            adj_matrix[x][y] = edge_weight + noise(noise_power)
            adj_matrix[y][x] = edge_weight + noise(noise_power)

    return adj_matrix