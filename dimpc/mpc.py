#!/usr/bin/python3

from copy import copy
from typing import List, Union

import casadi as cs

from dimpc.models import NonlinearQuadrotorModel
from dimpc.config import STATE_CONFIG, INPUT_CONFIG
from dimpc.util import get_default_vector, is_integer, subtract_lists


class MPC():
    def __init__(
        self,
        N: int,
        DT: float,
        Q: List[List[float]],
        R: List[List[float]],
        Qf: List[List[float]],
        model: NonlinearQuadrotorModel
    ) -> None:
        self._solver = self._init_minlp_solver(N, DT, Q, R, Qf, model)
        self._model = model
        self._N = N

    def solve(
        self,
        x0: List[float],
        xref: List[List[float]],
        lbx: List[float],
        ubx: List[float],
        lbu: List[float],
        ubu: List[float],
        xk_warmstart=None,
        uk_warmstart=None,
    ) -> dict:
        lbw = copy(x0)
        ubw = copy(x0)
        if xk_warmstart == None:
            xk_warmstart = (self._N + 1) * [copy(x0)]
        if uk_warmstart == None:
            uk_warmstart = self._N * \
                [get_default_vector(INPUT_CONFIG, self._model.u.shape[0])]
        warmstart = copy(subtract_lists(xk_warmstart[0], xref[0]))

        for k in range(self._N):
            lbw += copy(lbu + lbx)
            ubw += copy(ubu + ubx)
            warmstart += copy(
                uk_warmstart[k] + subtract_lists(xk_warmstart[k+1], xref[k+1])
            )

        lbg = self._N * self._model.x.shape[0] * [0.0]
        ubg = lbg
        return self._solver(x0=warmstart, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)


    def _get_objective(
        self,
        x: Union[cs.SX, cs.MX],
        u: Union[cs.SX, cs.MX],
        Q: List[List[float]],
        R,
    ) -> cs.SX:
        return x.T @ Q @ x + u.T @ R @ u

    def _get_terminal_objective(
        self,
        x: Union[cs.SX, cs.MX],
        Qf: List[List[float]],
    ) -> cs.SX:
        return x.T @ Qf @ x

    def _get_integrator(
        self,
        name: str,
        DT: float,
        L: cs.SX,
        model: NonlinearQuadrotorModel,
    ) -> cs.Function:
        f = cs.Function("f", [model.x, model.u], [model.xdot, L])
        x0 = cs.MX.sym("x_temp", model.x.shape)
        xf = x0
        u = cs.MX.sym("u_temp", model.u.shape)

        ORDER = 4
        cost = 0.0

        for i in range(ORDER):
            k1, k1_cost = f(xf, u)
            k2, k2_cost = f(xf + DT/2 * k1, u)
            k3, k3_cost = f(xf + DT/2 * k2, u)
            k4, k4_cost = f(xf + DT * k3, u)
            xf += DT/6 * (k1 +2*k2 +2*k3 +k4)
            cost += DT/6 * (k1_cost + 2*k2_cost + 2*k3_cost + k4_cost)

        return cs.Function(
            name,
            [x0, u], [xf, cost],
            ["x0", "u"], ["xf", "cost"]
        )

    def _init_minlp_solver(
        self,
        N: int,
        DT: float,
        Q: List[List[float]],
        R: List[List[float]],
        Qf: List[List[float]],
        model: NonlinearQuadrotorModel
    ) -> cs.nlpsol:
        L = self._get_objective(model.x, model.u, Q, R)
        F = self._get_integrator("F", DT, L, model)

        # New NLP variable for state
        x0 = cs.MX.sym("x0", model.x.shape)

        # Initialize parameter optimization inputs
        is_discrete = [False] * x0.shape[0]

        w = [x0]
        g = []
        J = 0.0

        # Formulate the MINLP
        xk = x0
        for k in range(N):
            # New MINLP variable for the control
            uk = cs.MX.sym("u" + str(k), model.u.shape)
            w += [uk]
            is_discrete += is_integer(INPUT_CONFIG, model.u.shape[0])

            # Integrate till the end of the interval
            Fk = F(x0=xk, u=uk)
            xf = Fk["xf"]
            J += Fk["cost"]

            # New MINLP variable for state at end of interval
            xk = cs.MX.sym("x" + str(k+1), xk.shape)
            w += [xk]
            is_discrete += [False] * xk.shape[0]

            # Add equality constraint
            g += [xf - xk]

        J += self._get_terminal_objective(xf, Qf)

        # Concatenate decision variables and constraint terms
        w = cs.vertcat(*w)
        g = cs.vertcat(*g)

        # Create a MINLP solver
        minlp_prob = {"f": J, "x": w, "g": g}
        return cs.nlpsol("nlp_solver", "bonmin", minlp_prob, {"discrete": is_discrete})
