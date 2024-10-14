from typing import List

import casadi as cs
import numpy as np

from par.utils.math import binomial_coefficient


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
    N: int,
) -> List[cs.SX]:
    J_inv = np.linalg.inv(J)
    angular_velocities = [angular_velocity_0]
    for k in range(1, N):
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
    N = len(angular_velocities)
    Hs = [cs.SX.eye(3)]
    for k in range(1, N):
        Hk = cs.SX.zeros(3)
        for n in range(k):
            Hk += binomial_coefficient(k-1, n) * \
                h(J, angular_velocities[n]) @ Hs[k-n-1]
        Hs += [Hk]
    return Hs
