from copy import copy
from typing import List, Tuple, Union

import casadi as cs
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib

from par.models import NonlinearQuadrotorModel, ParameterAffineQuadrotorModel
from par.config_utils import get_default_vector, get_subvector
from par.misc_utils import is_none, get_rk4_integrator
from par.config import STATE_CONFIG, INPUT_CONFIG
from par.constants import BIG_NEGATIVE, BIG_POSITIVE


# TODO: separate solver initialization and solver call
class MHPE():
    def __init__(
        self,
        dt: float,
        M: int,
        P: np.ndarray,
        S: np.ndarray,
        model: NonlinearQuadrotorModel
    ) -> None:
        self._dt = dt
        self._M = M
        self._P = P
        self._S = S
        self._model = model
        self._sol = {}

        self._xk = []
        self._uk = []
        self._wk = []
        self._theta = self._get_default_parameters()

    def get_parameter_estimate(self) -> np.ndarray:
        return self._theta

    def get_disturbance_estimates(self) -> List[np.ndarray]:
        return self._wk

    def plot_disturbances(self) -> None:
        """
        Display the series of control inputs
        and trajectory over prediction horizon.
        """
        fig, axs = plt.subplots(4, figsize=(11, 9))
        interp_N = 1000
        t = self._dt * np.arange(self._N)

        wk = np.array(self._wk)
        legend = ["x", "y", "z"]
        self._plot_trajectory(
            axs[0], t, wk[:, 0:3], interp_N, legend,
            "inertial frame\nvel noise"
        )
        legend = ["qw", "qx", "qy", "qz"]
        self._plot_trajectory(
            axs[1], t, wk[:, 3:7], interp_N, legend,
            "attitude rate noise"
        )
        legend = ["x", "y", "z"]
        self._plot_trajectory(
            axs[2], t, wk[:, 7:10], interp_N, legend,
            "body frame\naccel noise",
        )
        legend = ["x", "y", "z"]
        self._plot_trajectory(
            axs[3], t, wk[:, 10:13], interp_N, legend,
            "body frame\nang accel noise",
        )

        for ax in axs.flat:
            ax.set(xlabel="time (s)")
            ax.label_outer()
        plt.show()

    def solve(
        self,
        xM: np.ndarray,
        uM: np.ndarray,
        lb_theta=None,
        ub_theta=None,
        lbw=None,
        ubw=None,
        theta_ref=None,
        wk_ref=None,
        theta_warmstart=None,
        wk_warmstart=None,
    ) -> dict:
        # Update the trajectory history
        self._update_measurements(xM, uM)

        # Get discrete-time dynamics
        F = get_rk4_integrator(name="F", dt=self._dt, model=self._model)

        # New decision variable for parameters
        theta = cs.SX("theta", self._model.ntheta)

        # New constant for initial state
        x0 = cs.SX("x0", self._model.nx)

        # NLP variables
        d = [theta]             # Decision variables
        p = [x0]                # Constants
        g = []                  # Equality constraints
        lbg = []                # Inequality constraints
        ubg = []
        J = 0.0                 # Accumulated Cost

        # Get default disturbance reference values
        if is_none(wk_ref):
            wk_ref = self._wk

        # Formulate the NLP
        xk = x0
        for k in range(self._N):
            # New decision variable for process noise
            wk = cs.SX.sym("w" + str(k), self._model.nw)
            d += [wk]

            # New constant for control input
            uk = cs.SX.sym("u" + str(k), self._model.nu)
            p += [uk]

            # Get the state at the end of the time step
            xf = F(xk, uk, theta, wk)

            # New constant for state at end of interval
            xk = cs.SX.sym("x" + str(k+1), xk.shape)
            p += [xk]

            # Add dynamics equality constraint
            g += [xk - xf]
            lbg += xk.shape[0] * [0.0]
            ubg += xk.shape[0] * [0.0]

            # Add running cost
            J += self._get_stage_cost(w=wk, w_ref=wk_ref[k])

        # Get default parameter reference values
        if is_none(theta_ref):
            theta_ref = self._theta

        # Add parameter cost
        J += self._get_terminal_cost(theta=theta, theta_ref=theta_ref)

        # Concatenate decision variables and constraint terms
        d = cs.vertcat(*d)
        g = cs.vertcat(*g)

        # Create a  nlp  solver
        nlp_prob = {"f": J, "x": d, "p": p, "g": g, }
        solver = cs.nlpsol("nlp_solver", "ipopt", nlp_prob,)

        # Get default inequality constraints
        if is_none(lb_theta):
            lb_theta = self._get_default_lower_bound(shape=self._model.ntheta)
        if is_none(ub_theta):
            ub_theta = self._get_default_upper_bound(shape=self._model.ntheta)
        if is_none(lbw):
            lbw = self._get_default_lower_bound(shape=self._model.nw)
        if is_none(ubw):
            ubw = self._get_default_upper_bound(shape=self._model.nw)

        # Get default warmstart values
        if is_none(theta_warmstart):
            theta_warmstart = self._theta
        if is_none(wk_warmstart):
            wk_warmstart = self._wk

        # Build optimization arguments
        lbd = list(lb_theta)
        ubd = list(ub_theta)
        dref = list(theta_ref)
        warmstart = list(theta_warmstart)
        for k in range(self._M):
            lbd += list(lbw)
            ubd += list(ubw)
            dref += list(wk_ref[k])
            warmstart += list(wk_warmstart[k])

        # Solve and update estimates
        self._sol = solver(x0=warmstart, lbx=lbd, ubx=ubd, lbg=lbg, ubg=ubg)
        self._update_estimates()
        return self._sol

    def _update_measurements(
        self,
        xf: np.ndarray,
        uf: np.ndarray,
    ) -> None:
        self._xk += [xf]
        self._uk += [uf]
        self._wk += [np.zeros(self._model.nw)]
        if len(self._xk) < self._M \
        or len(self._uk) < self._M \
        or len(self._wk) < self._M:
            return
        else:
            self._xk.pop(0)
            self._uk.pop(0)
            self._wk.pop(0)

    def _update_estimates(self):
        self._theta = self._sol["x"][0]
        self._wk = []
        for i in range(self._M):
            wk = self._sol["x"][1 + i*self._model.nw : 1 + (i+1)*self._model.nw]
            self._wk += [np.array(wk)]

    def _get_default_lower_bound(
        self,
        shape: Union[int, Tuple[int, int]],
    ) -> np.ndarray:
        return BIG_NEGATIVE * np.ones(shape)

    def _get_default_upper_bound(
        self,
        shape: Union[int, Tuple[int, int]],
    ) -> np.ndarray:
        return BIG_POSITIVE * np.ones(shape)

    def _get_default_parameters(self) -> np.ndarray:
        if type(self._model) == ParameterAffineQuadrotorModel:
            return self._model.parameters.affine_vector
        else:
            return self._model.parameters.vector

    def _get_stage_cost(
        self,
        w: cs.SX,
        w_ref: np.ndarray,
    ) -> cs.SX:
        err = w - w_ref
        return err.T @ self._S @ err

    def _get_terminal_cost(
        self,
        theta: cs.SX,
        theta_ref: np.ndarray,
    ) -> cs.SX:
        err = theta - theta_ref
        return err.T @ self._P @ err

    def _plot_trajectory(
        self,
        ax: matplotlib.axes,
        Xs: np.ndarray,
        traj: np.ndarray,
        interp_N: int,
        legend: List[str],
        ylabel: str,
    ) -> None:
        ax.set_ylabel(ylabel)
        for i in range(traj.shape[1]):
            x_interp = self._get_interpolation(Xs, Xs, interp_N)
            y_interp = self._get_interpolation(Xs, traj[:, i], interp_N)
            ax.plot(x_interp, y_interp, label=legend[i])
        ax.legend()

    def _get_interpolation(
        self,
        Xs: np.ndarray,
        Ys: np.ndarray,
        N: int,
    ) -> np.ndarray:
        spline_func = make_interp_spline(Xs, Ys)
        interp_x = np.linspace(Xs.min(), Xs.max(), N)
        interp_y = spline_func(interp_x)
        return interp_y
