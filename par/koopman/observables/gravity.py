from typing import List

import casadi as cs

from par.utils.math import binomial_coefficient


def get_gs(
    g0: cs.SX,
    vs: List[cs.SX],
) -> List[cs.SX]:
    N = len(vs)
    gs = [g0]
    for k in range(1, N):
        gk = cs.SX.zeros(3)
        for n in range(k):
            gk += binomial_coefficient(k-1, n) * cs.skew(vs[n]).T @ gs[k-n-1]
        gs += [gk]
    return gs


def get_Gs(
    gs: List[cs.SX],
    vs: List[cs.SX],
    Hs: List[cs.SX],
    J: cs.SX,
) -> List[cs.SX]:
    assert len(gs) == len(vs) == len(Hs)
    N = len(gs)
    J_inv = cs.inv(J)
    Gs = [cs.SX.zeros((3,3))]
    for k in range(1, N):
        Gk = cs.SX.zeros((3,3))
        for n in range(k):
            Gk += binomial_coefficient(k-1, n) * \
                cs.skew(gs[n]).T @ J_inv @ Hs[k-n-1]
            Gk += binomial_coefficient(k-1, n) * \
                cs.skew(vs[n]).T @ J_inv @ Gs[k-n-1]
        Gs += [Gk]
    return Gs
