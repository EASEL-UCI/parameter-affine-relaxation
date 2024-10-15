from typing import List

import casadi as cs
import numpy as np

from par.utils.math import jordan_block
from par.koopman.observables import attitude, gravity, velocity


def get_nominal_state_matrix(
    N: int,
) -> np.ndarray:
    A1 = np.zeros((3*N, 3))
    A2 = np.eye(3 * (N-1))
    A3 = np.zeros((3, 3))
    return np.hstack(( A1, np.vstack((A2, A3)) ))

'''
def get_input_block(
    sub_blocks: List[cs.SX],
    i: int
) -> cs.SX:
    B = sub_blocks[0][i, :]
    Nv = len(sub_blocks)
    for j in range(1, Nv):
        B = cs.vertcat(B, sub_blocks[j][i, :])
    return B
'''

def get_attitude_state_matrix(N: int) -> cs.SX:
    return get_nominal_state_matrix(N)


def get_attitude_input_matrix(
    Hs: List[cs.SX],
    J: np.ndarray,
) -> cs.SX:
    J_inv = np.linalg.inv(J)
    N = len(Hs)
    B = cs.SX()
    for k in range(N):
        B = cs.vertcat(B, J_inv @ Hs[k])
    return B


def get_gravity_state_matrix(N: int) -> cs.SX:
    return get_nominal_state_matrix(N)


def get_gravity_input_matrix(
    Gs: List[cs.SX],
) -> cs.SX:
    N = len(Gs)
    B = cs.SX()
    for k in range(N):
        B = cs.vertcat(B, -Gs[k])
    return B
