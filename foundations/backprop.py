import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))
        error = y_hat - y_true
        dw = np.dot(((y_hat - y_true)* (y_hat*(1 - y_hat))), x)
        db = ((y_hat - y_true)* (y_hat*(1 - y_hat)))
        return tuple([np.round(dw, 5), np.round(db, 5)])