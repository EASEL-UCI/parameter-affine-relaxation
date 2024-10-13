from typing import List, Tuple

import casadi as cs
import numpy as np

from par.utils.math import binomial_coefficient
from par.utils.koopman.attitude import get_vs, get_Hs


def get_gs(
    g0: cs.SX,
    v0: cs.SX,
    J: np.ndarray,
    Ng: int,
) -> List[cs.SX]:
    vs = get_vs(v0, J, Ng)
    gs = [g0]
    for k in range(1, Ng):
        gk = cs.SX.zeros(3)
        for n in range(k):
            gk += binomial_coefficient(k-1, n) * cs.skew(vs[n]).T @ gs[k-n-1]
        gs += [gk]
    return gs


def get_Gs(
    g0: cs.SX,
    v0: cs.SX,
    J: np.ndarray,
    Ng: int,
) -> List[cs.SX]:
    inv_J = np.linalg.inv(J)
    vs = get_vs(v0, J, Ng)
    gs = get_gs(g0, v0, J, Ng)
    Hs = get_Hs(v0, J, Ng)
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
