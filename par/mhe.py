from typing import List, Union

import casadi as cs
import numpy as np

from par.dynamics.models import DynamicsModel, NonlinearQuadrotorModel, \
                                ParameterAffineQuadrotorModel
from par.dynamics.vectors import State, Input, ModelParameters, ProcessNoise, \
                                    AffineModelParameters, VectorList
from par.constants import BIG_NEGATIVE, BIG_POSITIVE
from par.config import PROCESS_NOISE_CONFIG, QP_SOLVER_CONFIG
from par.utils.config import get_config_values
from par.utils.misc import is_none


# TODO: separate solver initialization and solver call
class MHPE():
    def __init__(
        self,
        dt: float,
        M: int,
        P: np.ndarray,
        S: np.ndarray,
        model: DynamicsModel,
        x0: State = State(),
        plugin: str = "ipopt",
    ) -> None:
        assert type(model) == NonlinearQuadrotorModel \
            or type(model) == ParameterAffineQuadrotorModel
        self._dt = dt
        self._M = M
        self._P = P
        self._S = S
        self._model = model
        self._plugin = plugin

        self._sol = {}
        self._xs = VectorList(x0)
        self._us = VectorList()
        self._ws = VectorList()
        self._theta = self._model.parameters

        self._lbg = []
        self._ubg = []
        self._lba = BIG_NEGATIVE * np.ones(M * model.nx)
        self._uba = BIG_POSITIVE * np.ones(M * model.nx)
        self._lbw = ProcessNoise(get_config_values(
            "lower_bound", PROCESS_NOISE_CONFIG))
        self._ubw = ProcessNoise(get_config_values(
            "upper_bound", PROCESS_NOISE_CONFIG))

        if type(model) == ParameterAffineQuadrotorModel:
            self._lb_theta = AffineModelParameters(get_config_values(
                "lower_bound", model.parameters.config))
            self._ub_theta = AffineModelParameters(get_config_values(
                "upper_bound", model.parameters.config))
        else:
            self._lb_theta = ModelParameters(get_config_values(
                "lower_bound", model.parameters.config))
            self._ub_theta = ModelParameters(get_config_values(
                "upper_bound", model.parameters.config))
        self._solver = self._init_solver()

    def reset_measurements(self, x0: State) -> None:
        self._xs = VectorList(x0)
        self._ws = VectorList()

    def get_parameter_estimate(self) -> Union[ModelParameters, AffineModelParameters]:
        return self._theta

    def get_process_noise_estimates(self) -> VectorList:
        return self._ws

    def solve(
        self,
        xM: State,
        uM: Input,
        lb_theta: Union[ModelParameters, AffineModelParameters] = None,
        ub_theta: Union[ModelParameters, AffineModelParameters] = None,
        lbw: ProcessNoise = None,
        ubw: ProcessNoise = None,
        theta_guess: ModelParameters = None,
        ws_guess: VectorList = None,
    ) -> dict:
        # Update measurement history
        self._update_measurements(xM, uM)

        # Skip this solver call if measurement history isn't full length
        if not self._measurements_are_full():
            print("\nInput more measurements before solving!\n")
            return self._sol

        # Get default inequality constraints
        if is_none(lb_theta): lb_theta = self._lb_theta
        if is_none(ub_theta): ub_theta = self._ub_theta
        if is_none(lbw): lbw = self._lbw
        if is_none(ubw): ubw = self._ubw
        if is_none(theta_guess): theta_guess = self._theta
        if is_none(ws_guess): ws_guess = self._ws

        # Construct optimization arguments
        guess = theta_guess.as_list() + list(ws_guess.as_array().flatten())
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

    def _init_solver(self) -> dict:
        # Constant for state
        x0 = cs.SX.sym("x0", self._model.nx)
        # Decision variable for model parameters
        theta = cs.SX.sym("theta", self._model.ntheta)
        # Constant for parameter reference
        theta_ref = cs.SX.sym("theta_ref", self._model.ntheta)

        # Arguments for formulating NLP
        p = [x0, theta_ref]
        d = [theta]
        g = []
        lbg = []
        ubg = []
        J = self._get_parameter_cost(theta, theta_ref)

        # Formulate the NLP
        xk = x0
        for k in range(self._M):
            # New constant for control input
            uk = cs.SX.sym("u" + str(k), self._model.nu)
            p += [uk]

            # New decision variable for process noise
            wk = cs.SX.sym("w" + str(k), self._model.nw)
            d += [wk]

            # Get the state at the end of the time step
            xf = self._model.F(dt=self._dt, x=xk, u=uk, w=wk, theta=theta)

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
        opts = {"error_on_fail": False}
        nlp_prob = {"f": J, "x": d, "p": p, "g": g}
        if self._is_qp():
            return cs.qpsol("qp_solver", self._plugin, nlp_prob, opts)
        else:
            '''
            if self._plugin == "ipopt":
                opts = {"ipopt.max_iter": 3000, "ipopt.hessian_approximation": "exact"}
                if not is_verbose:
                    opts["ipopt.print_level"] = 0
                    opts["print_time"] = 0
                    opts["ipopt.sb"] = "yes"
            '''
            return cs.nlpsol("nlp_solver", self._plugin, nlp_prob, opts)

    def _measurements_are_full(
        self
    ) -> bool:
        if len(self._xs.get()) < self._M+1 or len(self._us.get()) < self._M \
        or len(self._ws.get()) < self._M:
            return False
        else:
            return True

    def _update_measurements(
        self,
        xM: State,
        uM: State,
    ) -> None:
        if self._measurements_are_full():
            self._xs.pop(0)
            self._us.pop(0)
            self._ws.pop(0)
        self._xs.append(xM)
        self._us.append(uM)
        self._ws.append(ProcessNoise())

    def _update_estimates(self):
        self._theta = ModelParameters(
            np.array(self._sol["x"][:self._model.ntheta]).flatten())
        ws = []
        for i in range(self._M):
            wk = self._sol["x"][self._model.ntheta + i*self._model.nw : \
                                self._model.ntheta + (i+1)*self._model.nw]
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

    def _is_qp(
        self,
    ) -> bool:
        try:
            QP_SOLVER_CONFIG[self._plugin]
            return True
        except KeyError:
            return False
        except TypeError:
            return False
