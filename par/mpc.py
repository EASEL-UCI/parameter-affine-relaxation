from copy import copy
from typing import List, Union

import casadi as cs
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib

from par.models import NonlinearQuadrotorModel
from par.config import STATE_CONFIG, INPUT_CONFIG
from par.config_utils import get_default_vector
from par.misc_utils import is_none


# TODO: separate solver initialization and solver call
class NMPC():
    def __init__(
        self,
        dt: float,
        N: int,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        model: NonlinearQuadrotorModel,
        xref=None,
        is_verbose=False,
    ) -> None:
        self._dt = dt
        self._N = N
        self._Q = Q
        self._R = R
        self._Qf = Qf
        self._model = model
        self._sol = {}

        self._lbg = None
        self._ubg = None
        self._lbx = get_default_vector("lower_bound", STATE_CONFIG)
        self._ubx = get_default_vector("upper_bound", STATE_CONFIG)
        self._lbu = get_default_vector("lower_bound", INPUT_CONFIG)
        self._ubu = get_default_vector("upper_bound", INPUT_CONFIG)
        self._uk_guess = N * [get_default_vector("default_value", INPUT_CONFIG)]
        self._solver = self._init_solver(xref, is_verbose)

    def get_state_trajectory(self) -> np.ndarray:
        nx = self._model.nx
        nu = self._model.nu
        state_traj = []
        for k in range(self._N + 1):
            state_traj += [
                np.array(self._sol["x"][k*(nx+nu) : k*(nx+nu) + nx]).flatten()
            ]
        return np.array(state_traj)

    def get_input_trajectory(self) -> np.ndarray:
        nx = self._model.nx
        nu = self._model.nu
        input_traj = []
        for k in range(self._N):
            input_traj += [np.array(
                self._sol["x"][(k+1)*nx+k*nu : (k+1)*nx+(k+1)*nu]
            ).flatten()]
        return np.array(input_traj)

    def plot_trajectory(
        self,
        xk=None,
        uk=None,
        dt=None,
        N=None,
    ) -> None:
        """
        Display the series of control inputs
        and trajectory over prediction horizon.
        """
        if is_none(dt):
            dt = self._dt
        if is_none(N):
            N = self._N

        t = dt * np.arange(N)
        interp_N = 1000
        fig, axs = plt.subplots(5, figsize=(11, 9))

        if is_none(uk):
            uk = np.array(self.get_input_trajectory())

        legend = ["u1", "u2", "u3", "u4"]
        self._plot_trajectory(
            axs[0], t, uk, interp_N, legend,
            "squared motor\nang vel (rad/s)^2",
        )

        if is_none(xk):
            xk = np.array(self.get_state_trajectory())
        if len(xk) > len(uk):
            xk = xk[:len(uk), :]

        legend = ["x", "y", "z"]
        self._plot_trajectory(
            axs[1], t, xk[:,:3], interp_N, legend,
            "pos (m)"
        )
        legend = ["qw", "qx", "qy", "qz"]
        self._plot_trajectory(
            axs[2], t, xk[:, 3:7], interp_N, legend,
            "att (quat)"
        )
        legend = ["vx", "vy", "vz"]
        self._plot_trajectory(
            axs[3], t, xk[:, 7:10], interp_N, legend,
            "body frame\nvel (m/s)"
        )
        legend = ["wx", "wy", "wz"]
        self._plot_trajectory(
            axs[4], t, xk[:, 10:13], interp_N, legend,
            "body frame\nang vel (rad/s)",
        )

        for ax in axs.flat:
            ax.set(xlabel="time (s)")
            ax.label_outer()
        plt.show()

    def solve(
        self,
        x: np.ndarray,
        theta=None,
        lbx=None,
        ubx=None,
        lbu=None,
        ubu=None,
        xk_guess=None,
        uk_guess=None,
    ) -> dict:
        # Get default inequality constraints
        if is_none(lbx):
            lbx = list(self._lbx)
        if is_none(ubx):
            ubx = list(self._ubx)
        if is_none(lbu):
            lbu = list(self._lbu)
        if is_none(ubu):
            ubu = list(self._ubu)

        # Get default warmstart values
        if is_none(xk_guess):
            xk_guess = (self._N + 1) * [x]
        if is_none(uk_guess):
            uk_guess = self._uk_guess

        lbd = list(x)
        ubd = list(x)
        guess = list(xk_guess[0])
        for k in range(self._N):
            lbd += list(lbu) + list(lbx)
            ubd += list(ubu) + list(ubx)
            guess += list(uk_guess[k]) + list(xk_guess[k+1])

        if is_none(theta):
            theta = self._model.get_default_parameter_vector()
        self._sol = self._solver(
            x0=guess, p=theta, lbx=lbd, ubx=ubd, lbg=self._lbg, ubg=self._ubg
        )
        return self._sol

    def _init_solver(
        self,
        xref: Union[None, np.ndarray],
        is_verbose: bool
    ) -> dict:
        # New decision variable for state
        x0 = cs.SX.sym("x0", self._model.nx)
        # New constant for parameters
        theta = cs.SX.sym("theta", self._model.ntheta)

        # Variables for formulating nlp
        d = [x0]
        p = [theta]
        g = []
        lbg = []
        ubg = []
        J = 0.0

        # Get default reference state
        if is_none(xref):
            xref = get_default_vector("default_value", STATE_CONFIG)

        # Formulate the nlp
        xk = x0
        for k in range(self._N):
            # New nlp variable for the control
            uk = cs.SX.sym("u" + str(k), self._model.nu)
            d += [uk]

            # Add running cost
            J += self._get_stage_cost(x=xk, u=uk, xref=xref)

            # Add terminal cost
            if k == self._N - 1:
                J += self._get_terminal_cost(x=xk, xref=xref)

            # Get the state at the end of the time step
            xf = self._model.F(dt=self._dt, x=xk, u=uk, theta=theta)

            # New NLP variable for state at end of interval
            xk = cs.SX.sym("x" + str(k+1), self._model.nx)
            d += [xk]

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
        if is_verbose:
            opts = {}
        else:
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        return cs.nlpsol("nlp_solver", "ipopt", nlp_prob, opts)

    def _get_stage_cost(
        self,
        x: cs.SX,
        u: cs.SX,
        xref: np.ndarray,
    ) -> cs.SX:
        err = x - xref
        return err.T @ self._Q @ err + u.T @ self._R @ u

    def _get_terminal_cost(
        self,
        x: cs.SX,
        xref: np.ndarray,
    ) -> cs.SX:
        err = x - xref
        return err.T @ (self._Qf - self._Q) @ err

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
