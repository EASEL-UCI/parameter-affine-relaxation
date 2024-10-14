from typing import List

import casadi as cs
import numpy as np

from par.utils.math import binomial_coefficient


def get_positions(
    position_0: cs.SX,
    angular_velocities: List[cs.SX]
) -> List[cs.SX]:
    N = len(angular_velocities)
    positions = [position_0]
    for k in range(1, N):
        position_k = cs.SX.zeros(3)
        for n in range(k):
            position_k += binomial_coefficient(k-1, n) * \
                        cs.skew(angular_velocities[n]).T @ positions[k-n-1]
        positions += [position_k]
    return positions


def get_Ps(
    positions: List[cs.SX],
    angular_velocities: List[cs.SX],
    Hs: List[cs.SX],
    J: np.ndarray,
) -> List[cs.SX]:
    assert len(positions) == len(angular_velocities) == len(Hs)
    N = len(positions)
    inv_J = np.linalg.inv(J)
    Ps = [cs.SX.zeros(3,3)]
    for k in range(1, N):
        Pk = cs.SX.zeros(3,3)
        for n in range(k):
            Pk += binomial_coefficient(k-1, n) * \
                cs.skew(positions[n]).T @ inv_J @ Hs[k-n-1]
            Pk += binomial_coefficient(k-1, n) * \
                cs.skew(angular_velocities[n]).T @ inv_J @ Ps[k-n-1]
        Ps += [Pk]
    return Ps
