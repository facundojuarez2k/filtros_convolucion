'''
    Repositorio de filtros
'''

import numpy as np

identity = np.array((
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
), dtype="int")

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

small_blur = np.ones((7, 7), dtype="float") * (1.0 / (7**2))

large_blur = np.ones((21, 21), dtype="float") * (1.0 / (21**2))

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

sobel_x = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobel_y = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

filters = {
    'identity': identity,
    'small_blur': small_blur,
    'large_blur': large_blur,
    'laplacian': laplacian,
    'sobel_x': sobel_x,
    'sobel_y': sobel_y,
}
