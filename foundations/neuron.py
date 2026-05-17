import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = np.dot(x, w) + b
        return np.round(1 / (1 + np.exp(-z)), 5) if activation == 'sigmoid' else np.round(np.maximum(0.0, z), 5)