"""
    Repositorio de filtros
"""

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

shift_substract = np.array((
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, -1]), dtype="int")

filters = {
    "identity": {
        "description": "Filtro identidad",
        "kernel": identity,
        "grayscale": False
    },
    "sharpen": {
        "description": "Agrega nitidez",
        "kernel": sharpen,
        "grayscale": False
    },
    "small_blur": {
        "description": "Blur gaussiano leve",
        "kernel": small_blur,
        "grayscale": False
    },
    "large_blur": {
        "description": "Blur gaussiano grande",
        "kernel": large_blur,
        "grayscale": False
    },
    "laplacian": {
        "description": "Filtro laplaciano",
        "kernel": laplacian,
        "grayscale": True
    },
    "sobel_x": {
        "description": "Filtro sobel horizontal",
        "kernel": sobel_x,
        "grayscale": True
    },
    "sobel_y": {
        "description": "Filtro sobel vertical",
        "kernel": sobel_y,
        "grayscale": True
    },
    "shift_substract": {
        "description": "Shift and substract",
        "kernel": shift_substract,
        "grayscale": True
    }
}
