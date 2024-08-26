#!/usr/bin/python3

from typing import List, Union

import casadi as cs
import numpy as np

from dimpc.models import NonlinearQuadrotorModel
from dimpc.config import INPUT_CONFIG


class DeliveryImplicitMPC():
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

    def solve(
        self,
        x: List[float],
        u: List[float],
        lbx: List[float],
        ubx: List[float],
        lbu: List[float],
        ubu: List[float],
        x_guess=None,
        u_guess=None,
    ):
        lbw = []
        ubw = []

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
        w = [x0]
        g = []
        is_integer = [False] * model.x.shape[0]
        J = 0.0

        # Formulate the MINLP
        xk = x0
        for k in range(N):
            # New MINLP variable for the control
            uk = cs.MX.sym("u" + str(k), model.u.shape)
            w += [uk]
            for sub_config in INPUT_CONFIG.values():
                is_int = [False]
                if sub_config["type"] == int or sub_config["member_type"] == int:
                    is_int = [True]
                is_integer += is_int * sub_config["dimensions"]

            # Integrate till the end of the interval
            Fk = F(x0=xk, u=uk)
            xf = Fk["xf"]
            J += Fk["cost"]

            # New MINLP variable for state at end of interval
            xk = cs.MX.sym("x" + str(k+1), model.x.shape)
            w += [xk]
            is_integer += [False] * model.x.shape[0]

            # Add equality constraint
            g += [xf - xk]

        J += self._get_terminal_objective(xf, Qf)

        # Concatenate decision variables and constraint terms
        w = cs.vertcat(*w)
        g = cs.vertcat(*g)

        # Create a MINLP solver
        minlp_prob = {"f": J, "x": w, "g": g}
        return cs.nlpsol("nlp_solver", "bonmin", minlp_prob, {"discrete": is_integer})
