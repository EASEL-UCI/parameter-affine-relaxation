from typing import List

import casadi as cs
import numpy as np

from par.utils.math import binomial_coefficient


def get_k_gs(
    g0: cs.SX,
    vs: List[cs.SX],
) -> List[cs.SX]:
    Ng = len(vs)
    gs = [g0]
    for k in range(1, Ng):
        gk = cs.SX.zeros(3)
        for n in range(k):
            gk += binomial_coefficient(k-1, n) * cs.skew(vs[n]).T @ gs[k-n-1]
        gs += [gk]
    return gs


def get_k_Gs(
    gs: List[cs.SX],
    vs: List[cs.SX],
    Hs: List[cs.SX],
    J: np.ndarray,
) -> List[cs.SX]:
    assert len(gs) == len(vs) == len(Hs)
    Ng = len(gs)
    inv_J = np.linalg.inv(J)
    Gs = [cs.SX.zeros(3,3)]
    for k in range(1, Ng):
        Gk = cs.SX.zeros(3,3)
        for n in range(k):
            Gk += binomial_coefficient(k-1, n) * \
                    cs.skew(gs[n]).T @ inv_J @ Hs[k-n-1]
            Gk += binomial_coefficient(k-1, n) * \
                    cs.skew(vs[n]).T @ inv_J @ Gs[k-n-1]
        Gs += [Gk]
    return Gs
