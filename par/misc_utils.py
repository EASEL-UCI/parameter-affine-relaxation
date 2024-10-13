from typing import Any, Tuple, Union
import numpy as np


def is_none(a: Any) -> bool:
    if type(a) == type(None):
        return True
    else:
        return False


def alternating_ones(shape: Union[int, Tuple[int]]) -> np.ndarray:
    ones = np.ones(shape)
    ones[::2] = -1.0
    return ones


def jordan_block(lam: float, n: int) -> np.ndarray:
    J = np.diag(lam * np.ones(n))
    for i in range(n - 1):
        J[i, i+1] = 1.0
    return J
