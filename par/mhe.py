from typing import List, Union

import casadi as cs
import numpy as np

from par.dynamics.models import DynamicsModel, NonlinearQuadrotorModel, \
                                ParameterAffineQuadrotorModel
from par.dynamics.vectors import State, Input, ModelParameters, ProcessNoise, \
                                    VectorList
from par.config import NOISE_CONFIG
from par.utils.config import get_config_values, get_dimensions
from par.utils.misc import is_none


# TODO: separate solver initialization and solver call
class MHPE():
    def __init__(
        self,
        dt: float,
        M: int,
        P: np.ndarray,
        S: np.ndarray,
        x0: State,
        model: DynamicsModel,
        is_verbose=False,
    ) -> None:
        assert type(model) == NonlinearQuadrotorModel \
            or type(model) == ParameterAffineQuadrotorModel
        self._dt = dt
        self._M = M
        self._P = P
        self._S = S
        self._S = S
        self._model = model

        self._sol = {}
        self._xs = VectorList(x0)
        self._us = VectorList()
        self._ws = VectorList()
        self._theta = ModelParameters(self._model.get_default_parameter_vector())

        self._lbg = []
        self._ubg = []
        self._lbw = ProcessNoise(get_config_values("lower_bound", NOISE_CONFIG))
        self._ubw = ProcessNoise(get_config_values("upper_bound", NOISE_CONFIG))
        self._lb_theta = ModelParameters(get_config_values(
            "lower_bound", model.parameter_config))
        self._ub_theta = ModelParameters(get_config_values(
            "upper_bound", model.parameter_config))
        self._solver = self._init_solver(is_verbose)

    def get_state_estimates(self) -> VectorList:
        return self._xs

    def get_process_noise_estimates(self) -> VectorList:
        return self._ws

    def solve(
        self,
        xM: State,
        uM: Input,
        lb_theta: ModelParameters = None,
        ub_theta: ModelParameters = None,
        lbw: ProcessNoise = None,
        ubw: ProcessNoise = None,
        theta_guess: ModelParameters = None,
        ws_guess: ProcessNoise = None,
    ) -> dict:
        # Update measurement history
        self._update_measurements(xM, uM)

        # Get default inequality constraints
        if is_none(lb_theta): lb_theta = self._lb_theta
        if is_none(ub_theta): ub_theta = self._ub_theta
        if is_none(lbw): lbw = self._lbw
        if is_none(ubw): ubw = self._ubw
        if is_none(theta_guess): theta_guess = self._theta
        if is_none(ws_guess): ws_guess = self._ws

        # Construct optimization arguments
        guess = theta_guess.as_list() + ws_guess.as_list()
        lbd = lb_theta.as_list()
        ubd = ub_theta.as_list()
        p = self._xs.get(0).as_list() + self._theta.as_list()
        for k in range(self._M):
            lbd += lbw.as_list()
            ubd += ubw.as_list()
            p += self._us.get(k).as_list() + self._xs.get(k + 1).as_list() \
                    + self._ws.get(k).as_list()

        # Solve
        self._sol = self._solver(
            x0=guess, p=p, lbx=lbd, ubx=ubd, lbg=self._lbg, ubg=self._ubg)
        self._update_estimates()
        return self._sol

    def _init_solver(
        self,
        is_verbose: bool
    ) -> dict:
        # Decision variable for state
        x0 = cs.SX.sym("x0", self._model.nx)
        # Constant for model parameters
        theta = cs.SX.sym("theta", self._model.ntheta)
        # Constant for parameter estimate reference
        theta_ref = cs.SX.sym("theta_ref", self._model.ntheta)

        # Variables for formulating NLP
        p = [x0, theta_ref]
        d = [theta]
        g = []
        lbg = []
        ubg = []
        J = self._get_parameter_cost(theta, theta_ref)

        # Formulate the NLP
        xk = x0
        for k in range(self._M):
            # New decision variable for process noise
            wk = cs.SX.sym("w" + str(k), self._model.nw)
            d += [wk]

            # New constant for control input
            uk = cs.SX.sym("u" + str(k), self._model.nu)
            p += [uk]

            # Get the state at the end of the time step
            xf = self._model.F(dt=self._dt, x=xk, u=uk, theta=theta)

            # New constant for the state at the end of the interval
            xk = cs.SX.sym("x" + str(k+1), self._model.nx)
            p += [xk]

            # New constant for disturbance reference tracking
            wref_k = cs.SX.sym("wref" + str(k), self._model.nw)
            p += [wref_k]

            # Add running cost
            J += self._get_stage_cost(wk, wref_k)

            # Add dynamics equality constraint
            g += [xf - xk]
            lbg += self._model.nx * [0.0]
            ubg += self._model.nx * [0.0]

        # Concatenate decision variables and constraint terms
        d = cs.vertcat(*d)
        p = cs.vertcat(*p)
        g = cs.vertcat(*g)

        # Initialize equality constraint values
        self._lbg = lbg
        self._ubg = ubg

        # Create NLP solver
        nlp_prob = {"f": J, "x": d, "p": p, "g": g}
        opts = {"ipopt.max_iter": 3000} #{"ipopt.hessian_approximation": "exact"}
        if not is_verbose:
            opts["ipopt.print_level"] = 0
            opts["print_time"] = 0
            opts["ipopt.sb"] = "yes"
            #opts["ipopt.hessian_approximation"] = "exact"
        #opts = {"error_on_fail": False}
        #return cs.qpsol("nlp_solver", "osqp", nlp_prob, opts)
        return cs.nlpsol("nlp_solver", "ipopt", nlp_prob, opts)

    def _update_measurements(
        self,
        xM: State,
        uM: State,
    ) -> None:
        self._xs.append(xM)
        self._us.append(uM)
        self._ws.append(ProcessNoise(np.zeros(self._model.nw)))
        if len(self._xs.get()) < self._M + 1 or len(self._us.get()) < self._M \
        or len(self._ws.get()) < self._M:
            return
        else:
            self._xs.pop(0)
            self._us.pop(0)
            self._ws.pop(0)

    def _update_estimates(self):
        self._theta = ModelParameters(np.array(self._sol["x"][0]).flatten())
        ws = []
        for i in range(self._M):
            wk = self._sol["x"][1 + i*self._model.nw : 1 + (i+1)*self._model.nw]
            ws += [ProcessNoise(np.array(wk).flatten())]
        self._ws = VectorList(ws)

    def _get_stage_cost(
        self,
        w: cs.SX,
        wref: cs.SX,
    ) -> cs.SX:
        err = w - wref
        return err.T @ self._S @ err

    def _get_parameter_cost(
        self,
        theta: cs.SX,
        theta_ref: np.ndarray,
    ) -> cs.SX:
        err = theta - theta_ref
        return err.T @ self._P @ err
