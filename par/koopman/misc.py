from typing import List

import casadi as cs
import numpy as np

from par.utils.math import jordan_block


def get_state_matrix(
    N: int,
) -> np.ndarray:
    return jordan_block(lam=0.0, n=int(3*N))


def get_input_block(
    sub_blocks: List[cs.SX],
    i: int
) -> cs.SX:
    B = sub_blocks[0][i, :]
    Nv = len(sub_blocks)
    for j in range(1, Nv):
        B = cs.vertcat(B, sub_blocks[j][i, :])
    return B
