from typing import List

import casadi as cs
import numpy as np

from par.utils.math import binomial_coefficient


def get_velocities(
    velocity_0: cs.SX,
    angular_velocities: List[cs.SX],
) -> List[cs.SX]:
    velocities = [velocity_0]
    N = len(angular_velocities)
    for k in range(1, N):
        velocity_k = cs.SX.zeros(3)
        for n in range(k):
            velocity_k += binomial_coefficient(k-1, n) * \
                        cs.skew(angular_velocities[n]).T @ velocities[k-n-1]
        velocities += [velocity_k]
    return velocities


def get_omegas(
    velocities: List[cs.SX]
) -> List[cs.SX]:
    omegas = [cs.SX.eye(3)]
    N = len(velocities)
    for k in range(N):
        omega_k = cs.SX.eye(3)
        for n in range(k):
            omega_k += binomial_coefficient(k-1, n) * \
                cs.skew(velocities[n-1]).T @ omegas[k-n-1]
        omegas += [omega_k]
    return omegas


def get_Vs(
    velocities: List[cs.SX],
    angular_velocities: List[cs.SX],
    Hs: List[cs.SX],
    J: np.ndarray,
) -> List[cs.SX]:
    assert len(velocities) == len(angular_velocities) == len(Hs)
    N = len(velocities)
    inv_J = np.linalg.inv(J)
    Vs = [cs.SX.zeros(3,3)]
    for k in range(1, N):
        Vk = cs.SX.zeros(3,3)
        for n in range(k):
            Vk += binomial_coefficient(k-1, n) * \
                cs.skew(velocities[n]).T @ inv_J @ Hs[k-n-1]
            Vk += binomial_coefficient(k-1, n) * \
                cs.skew(angular_velocities[n]).T @ inv_J @ Vs[k-n-1]
        Vs += [Vk]
    return Vs
