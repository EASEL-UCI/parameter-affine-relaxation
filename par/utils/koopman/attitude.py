from typing import List

import casadi as cs
import numpy as np

from par.utils.math import binomial_coefficient
from par.utils.koopman.misc import get_state_matrix, get_input_block


def gamma(
    J: np.ndarray,
    v: cs.SX,
) -> cs.SX:
    return J @ v


def h(
    J: np.ndarray,
    v: cs.SX,
) -> cs.SX:
    return cs.skew(gamma(J, v)) @ np.linalg.inv(J) - cs.skew(v)


def get_Vs(
    v0: cs.SX,
    J: np.ndarray,
    Nv: int,
) -> List[cs.SX]:
    J_inv = np.linalg.inv(J)
    Vs = [v0]
    for k in range(1, Nv):
        summation = cs.SX.zeros(3)
        for n in range(k):
            gamma_n = gamma(J, Vs[n])
            summation += binomial_coefficient(k-1, n) * \
                            cs.skew(gamma_n) @ Vs[k-n-1]
        vk = J_inv @ summation
        Vs += [vk]
    return Vs


def get_Hs(
    v0: cs.SX,
    J: np.ndarray,
    Nv: int,
) -> List[cs.SX]:
    Vs = get_Vs(v0, J, Nv)
    Hs = [cs.SX.eye(3)]
    for k in range(1, Nv):
        Hk = cs.SX.zeros(3)
        for n in range(k):
            Hk += binomial_coefficient(k-1, n) * h(J, Vs[n]) @ Hs[k-n-1]
        Hs += [Hk]
    return Hs


def get_attitude_input_matrix(
    v0: cs.SX,
    J: np.ndarray,
    Nv: int,
) -> cs.SX:
    J_inv = np.linalg.inv(J)

    sub_blocks = get_Hs(v0, J, Nv)
    for i in range(len(sub_blocks)):
        sub_blocks[i] = J_inv @ sub_blocks[i]

    B = cs.SX()
    for i in range(3):
        cs.vertcat(B, get_input_block(sub_blocks, i))
    return B


def get_attitude_state_matrix(Nv: int) -> np.ndarray:
    return get_state_matrix(Nv)