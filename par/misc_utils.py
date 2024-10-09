from typing import Any, Tuple, Union
import numpy as np


def is_none(a: Any) -> bool:
    if type(a) == type(None):
        return True
    else:
        return False


def get_alternating_ones(shape: Union[int, Tuple[int]]) -> np.ndarray:
    ones = np.ones(shape)
    ones[::2] = -1.0
    return ones
