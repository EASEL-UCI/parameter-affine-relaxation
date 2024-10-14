from typing import List

import casadi as cs
import numpy as np

from par.utils.math import binomial_coefficient
from par.koopman.misc import get_state_matrix, get_input_block


def gamma(
    J: np.ndarray,
    angular_velocity: cs.SX,
) -> cs.SX:
    return J @ angular_velocity


def h(
    J: np.ndarray,
    angular_velocity: cs.SX,
) -> cs.SX:
    return cs.skew(gamma(J, angular_velocity)) @ np.linalg.inv(J) \
        - cs.skew(angular_velocity)


def get_angular_velocities(
    angular_velocity_0: cs.SX,
    J: np.ndarray,
    Nv: int,
) -> List[cs.SX]:
    J_inv = np.linalg.inv(J)
    angular_velocities = [angular_velocity_0]
    for k in range(1, Nv):
        summation = cs.SX.zeros(3)
        for n in range(k):
            gamma_n = gamma(J, angular_velocities[n])
            summation += binomial_coefficient(k-1, n) * \
                            cs.skew(gamma_n) @ angular_velocities[k-n-1]
        angular_velocity_k = J_inv @ summation
        angular_velocities += [angular_velocity_k]
    return angular_velocities


def get_Hs(
    angular_velocities: List[cs.SX],
    J: np.ndarray,
) -> List[cs.SX]:
    Nv = len(angular_velocities)
    Hs = [cs.SX.eye(3)]
    for k in range(1, Nv):
        Hk = cs.SX.zeros(3)
        for n in range(k):
            Hk += binomial_coefficient(k-1, n) * \
                h(J, angular_velocities[n]) @ Hs[k-n-1]
        Hs += [Hk]
    return Hs


def get_input_matrix(
    Hs: List[cs.SX],
    J: np.ndarray,
) -> cs.SX:
    Nv = len(Hs)
    J_inv = np.linalg.inv(J)
    for i in range(Nv):
        Hs[i] = J_inv @ Hs[i]
    B = cs.SX()
    for i in range(3):
        B = cs.vertcat(B, get_input_block(Hs, i))
    return B


def get_attitude_state_matrix(Nv: int) -> np.ndarray:
    return get_state_matrix(Nv)
