from typing import Union, Callable, Tuple

import casadi as cs
import numpy as np

from par.utils import quat
from par.dynamics.vectors import ModelParameters
from par.koopman.dynamics import get_state_matrix, get_input_matrix
from par.koopman.observables import attitude, gravity, velocity, position
from par.constants import GRAVITY
from par.utils.misc import is_none, alternating_ones, convert_casadi_to_numpy_vector
from par.utils.config import symbolic, get_dimensions, get_subvector, \
                                insert_subvector
from par.config import PARAMETER_CONFIG, RELAXED_PARAMETER_CONFIG,STATE_CONFIG, \
                        KOOPMAN_CONFIG, INPUT_CONFIG, NOISE_CONFIG


class DynamicsModel():
    def __init__(
        self,
        parameters,
    ) -> None:
        self._parameters = parameters
        self._f = None
        self._ntheta = None
        self._parameter_config = None
        self._order = 1
        self._nx = get_dimensions(STATE_CONFIG)
        self._nu = get_dimensions(INPUT_CONFIG)
        self._nw = get_dimensions(NOISE_CONFIG)

    @property
    def parameters(self) -> ModelParameters:
        return self._parameters

    @parameters.setter
    def parameters(
        self,
        parameters: ModelParameters
    ) -> None:
        self._parameters = parameters

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._nu

    @property
    def nw(self) -> int:
        return self._nw

    @property
    def ntheta(self) -> int:
        return self._ntheta

    @property
    def parameter_config(self) -> dict:
        return self._parameter_config

    @property
    def order(self)-> int:
        return self._order

    def F(
        self,
        dt: float,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w=None,
        theta=None,
    ) -> Union[np.ndarray, cs.SX]:
        if is_none(theta):
            theta = self.get_default_parameter_vector()
        if is_none(w):
            w = np.zeros(self.nw)
        xf = self.rk4(self.f, dt, x, u, w, theta)
        if type(xf) == cs.DM:
            return np.array(xf).flatten()
        else:
            return xf

    def f(
        self,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w=None,
        theta=None,
    ) -> Union[np.ndarray, cs.SX]:
        if is_none(self._f):
            raise Exception("Model has no continuous-time dynamics!")
        if is_none(theta):
            theta = self.get_default_parameter_vector()
        if is_none(w):
            w = np.zeros(self.nw)
        return self._f(x, u, w, theta)

    def rk4(
        self,
        f: Callable,
        dt: float,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX],
        theta: Union[np.ndarray, cs.SX],
    ) -> Union[np.ndarray, cs.SX]:
        k1 = f(x, u, w, theta)
        k2 = f(x + dt/2 * k1, u, w, theta)
        k3 = f(x + dt/2 * k2, u, w, theta)
        k4 = f(x + dt * k3, u, w, theta)
        return x + dt/6 * (k1 +2*k2 +2*k3 +k4)

    def check_parameters(self) -> None:
        if is_none(self._parameters):
            raise Exception("Model has no parameter values!")

    def get_default_parameter_vector(self) -> None:
        raise NotImplementedError("Parameter vector getter not implemented!")

    def get_substates(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray]:
        p = get_subvector(x, "POSITION", STATE_CONFIG)
        q = get_subvector(x, "ATTITUDE", STATE_CONFIG)
        v = get_subvector(x, "BODY_LINEAR_VELOCITY", STATE_CONFIG)
        w = get_subvector(x, "BODY_ANGULAR_VELOCITY", STATE_CONFIG)
        return p, q, v, w


class NonlinearQuadrotorModel(DynamicsModel):
    def __init__(
        self,
        parameters=None,
    ) -> None:
        super().__init__(parameters)
        self._parameter_config = PARAMETER_CONFIG
        self._ntheta = get_dimensions(PARAMETER_CONFIG)
        self._set_model()

    def get_default_parameter_vector(self) -> np.ndarray:
        super().check_parameters()
        return self._parameters.vector

    def _set_model(self) -> None:
        p = symbolic("POSITION", STATE_CONFIG)
        q = symbolic("ATTITUDE", STATE_CONFIG)
        vB = symbolic("BODY_LINEAR_VELOCITY", STATE_CONFIG)
        wB = symbolic("BODY_ANGULAR_VELOCITY", STATE_CONFIG)
        x = cs.SX(cs.vertcat(p, q, vB, wB))

        m = symbolic("m", PARAMETER_CONFIG)
        a = symbolic("a", PARAMETER_CONFIG)
        Ixx = symbolic("Ixx", PARAMETER_CONFIG)
        Iyy = symbolic("Iyy", PARAMETER_CONFIG)
        Izz = symbolic("Izz", PARAMETER_CONFIG)
        k = symbolic("k", PARAMETER_CONFIG)
        c = symbolic("c", PARAMETER_CONFIG)
        r = symbolic("r", PARAMETER_CONFIG)
        s = symbolic("s", PARAMETER_CONFIG)
        theta = cs.SX(cs.vertcat(m, a, Ixx, Iyy, Izz, k, c, r, s))

        # Constants
        g = cs.SX(cs.vertcat(0, 0, -GRAVITY))
        A = cs.SX(cs.diag(a))
        J = cs.SX(cs.diag(cs.vertcat(Ixx, Iyy, Izz)))

        # Control input terms
        u = symbolic("MOTOR_SPEED_SQUARED", INPUT_CONFIG)
        K = cs.SX(cs.vertcat(
            cs.SX.zeros(2, self.nu),
            k.T,
        ))
        B = cs.SX(cs.vertcat(
            (k * s).T,
            -(k * r).T,
            cs.SX(alternating_ones(self.nu)).T * c.T,
        ))

        # Additive process noise
        w = cs.SX.sym("w", self.nw)

        # Continuous-time dynamics
        xdot = w + cs.SX(cs.vertcat(
            quat.Q(q)@vB,
            0.5 * quat.G(q)@wB,
            quat.Q(q).T@g + (K@u - A@vB) / m - cs.cross(wB, vB),
            cs.inv(J) @ (B@u - cs.cross(wB, J@wB))
        ))

        # Define dynamics function
        self._f = cs.Function(
            "f_NonlinearQuadrotorModel",
            [x, u, w, theta], [xdot]
        )


class ParameterAffineQuadrotorModel(DynamicsModel):
    def __init__(
        self,
        parameters=None,
    ) -> None:
        super().__init__(parameters)
        self._parameter_config = RELAXED_PARAMETER_CONFIG
        self._ntheta = get_dimensions(RELAXED_PARAMETER_CONFIG)
        self._set_affine_model()

    def get_default_parameter_vector(self) -> np.ndarray:
        super().check_parameters()
        return self._parameters.affine_vector

    def _set_affine_model(self) -> None:
        p = symbolic("POSITION", STATE_CONFIG)
        q = symbolic("ATTITUDE", STATE_CONFIG)
        vB = symbolic("BODY_LINEAR_VELOCITY", STATE_CONFIG)
        wB = symbolic("BODY_ANGULAR_VELOCITY", STATE_CONFIG,)
        x = cs.SX(cs.vertcat(p, q, vB, wB))

        g = cs.vertcat(0, 0, -GRAVITY)

        # parameter-independent dynamics
        F = cs.SX(cs.vertcat(
            quat.Q(q) @ vB,
            0.5 * quat.G(q) @ wB,
            quat.Q(q).T @ g - cs.cross(wB, vB),
            cs.SX.zeros(3),
        ))

        u = symbolic("MOTOR_SPEED_SQUARED", INPUT_CONFIG)
        K = cs.SX(cs.vertcat(
            cs.SX.zeros(2, self.nu),
            u.T,
        ))
        A = cs.SX(cs.diag(vB))
        C = cs.SX(cs.vertcat(
            cs.horzcat( u.T, cs.SX.zeros(1,2*self.nu) ),
            cs.horzcat(cs.SX.zeros(1,self.nu), -u.T, cs.SX.zeros(1,self.nu) ),
            cs.horzcat(
                cs.SX.zeros(1,2*self.nu),
                cs.SX(alternating_ones(self.nu)).T * u.T
            ),
        ))
        I = cs.SX(cs.diag(cs.vertcat(wB[1]*wB[2], wB[0]*wB[2], wB[0]*wB[1])))

        # Parameter-coupled dynamics
        G = cs.SX(cs.vertcat(
            cs.SX.zeros(7, 6 + 4*self.nu),
            cs.horzcat( -A, K, cs.SX.zeros(3, 3 + 3*self.nu) ),
            cs.horzcat( cs.SX.zeros(3, 3 + self.nu), C, -I ),
        ))

        # Additive process noise
        w = cs.SX.sym("w", self.nw)

        # Affine parameters
        theta = cs.SX.sym("relaxed_parameters", self.ntheta)

        # Continuous-time dynamics
        xdot = w + F + G @ theta

        # Define dynamics function
        self._f = cs.Function(
            "f_ParameterAffineQuadrotorModel",
            [x, u, w, theta], [xdot]
        )


class KoopmanLiftedQuadrotorModel(DynamicsModel):
    def __init__(
        self,
        observables_order: int,
        parameters=None,
    ) -> None:
        super().__init__(parameters)
        self._parameter_config = PARAMETER_CONFIG
        self._ntheta = get_dimensions(PARAMETER_CONFIG)
        self._nx = observables_order * get_dimensions(KOOPMAN_CONFIG)
        self._nw = self._nx
        self._order = observables_order
        self._set_lifted_model()

    def get_default_parameter_vector(self) -> np.ndarray:
        super().check_parameters()
        return self._parameters.vector

    def F(
        self,
        dt: float,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        x_init: Union[np.ndarray, cs.SX],
        w=None,
        theta=None,
    ) -> Union[np.ndarray, cs.SX]:
        if is_none(theta):
            theta = self.get_default_parameter_vector()
        if is_none(w):
            w = np.zeros(self.nw)
        xf = self.rk4(self.f, dt, x, u, w, theta, x_init)
        if type(xf) == cs.DM:
            return np.array(xf).flatten()
        else:
            return xf

    def f(
        self,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        x_init: Union[np.ndarray, cs.SX],
        w=None,
        theta=None,
    ) -> Union[np.ndarray, cs.SX]:
        if is_none(self._f):
            raise Exception("Model has no continuous-time dynamics!")
        if is_none(theta):
            theta = self.get_default_parameter_vector()
        if is_none(w):
            w = np.zeros(self.nw)
        return self._f(x, u, w, theta, x_init)

    def rk4(
        self,
        f: Callable,
        dt: float,
        x: Union[np.ndarray, cs.SX],
        u: Union[np.ndarray, cs.SX],
        w: Union[np.ndarray, cs.SX],
        theta: Union[np.ndarray, cs.SX],
        x_init: Union[np.ndarray, cs.SX],
    ) -> Union[np.ndarray, cs.SX]:
        k1 = f(x, u, x_init, w, theta)
        k2 = f(x + dt/2 * k1, u, x_init, w, theta)
        k3 = f(x + dt/2 * k2, u, x_init, w, theta)
        k4 = f(x + dt * k3, u, x_init, w, theta)
        return x + dt/6 * (k1 +2*k2 +2*k3 +k4)

    def convert_nominal_to_koopman_lifted_state(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        p, q, v, w = self.get_substates(x)

        rot = quat.Q(q)
        p_body = rot.T @ p
        g_body = rot.T @ np.array([0, 0, -GRAVITY])
        theta = self.get_default_parameter_vector()
        J = np.diag(np.hstack((
            get_subvector(theta, "Ixx", PARAMETER_CONFIG),
            get_subvector(theta, "Iyy", PARAMETER_CONFIG),
            get_subvector(theta, "Izz", PARAMETER_CONFIG),
        )))

        ws = attitude.get_ws(w, J, self.order)
        ps = position.get_ps(p_body, ws)
        vs = velocity.get_vs(v, ws)
        gs = gravity.get_gs(g_body, ws)

        ws_stacked = convert_casadi_to_numpy_vector(cs.vertcat(*ws))
        ps_stacked = convert_casadi_to_numpy_vector(cs.vertcat(*ps))
        vs_stacked = convert_casadi_to_numpy_vector(cs.vertcat(*vs))
        gs_stacked = convert_casadi_to_numpy_vector(cs.vertcat(*gs))

        x_lifted = np.zeros(get_dimensions(KOOPMAN_CONFIG, copies=self.order))
        x_lifted = insert_subvector(x_lifted, ps_stacked,
            "BODY_FRAME_POSITION", KOOPMAN_CONFIG, copies=self.order)
        x_lifted = insert_subvector(x_lifted, vs_stacked,
            "BODY_FRAME_LINEAR_VELOCITY", KOOPMAN_CONFIG, copies=self.order)
        x_lifted = insert_subvector(x_lifted, gs_stacked,
            "BODY_FRAME_GRAVITY", KOOPMAN_CONFIG, copies=self.order)
        x_lifted = insert_subvector(x_lifted, ws_stacked,
            "BODY_FRAME_ANGULAR_VELOCITY", KOOPMAN_CONFIG, copies=self.order)
        return x_lifted

    def convert_nominal_to_koopman_initialization_state(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        p, q, v, w = self.get_substates(x)

        rot = quat.Q(q)
        p_body = rot.T @ p
        g_body= rot.T @ np.array([0, 0, -GRAVITY])

        x_init = np.zeros(get_dimensions(KOOPMAN_CONFIG))
        x_init = insert_subvector(
            x_init, p_body, "BODY_FRAME_POSITION", KOOPMAN_CONFIG)
        x_init = insert_subvector(
            x_init, v, "BODY_FRAME_LINEAR_VELOCITY", KOOPMAN_CONFIG)
        x_init = insert_subvector(
            x_init, g_body, "BODY_FRAME_GRAVITY", KOOPMAN_CONFIG)
        x_init = insert_subvector(
            x_init, w, "BODY_FRAME_ANGULAR_VELOCITY", KOOPMAN_CONFIG)
        return x_init

    def _set_lifted_model(self) -> None:
        pB_init = symbolic("BODY_FRAME_POSITION", KOOPMAN_CONFIG)
        vB_init = symbolic("BODY_FRAME_LINEAR_VELOCITY", KOOPMAN_CONFIG)
        gB_init = symbolic("BODY_FRAME_GRAVITY", KOOPMAN_CONFIG)
        wB_init = symbolic("BODY_FRAME_ANGULAR_VELOCITY", KOOPMAN_CONFIG)
        x_init = cs.vertcat(pB_init, vB_init, gB_init, wB_init)

        pB = symbolic("BODY_FRAME_POSITION", KOOPMAN_CONFIG, self._order)
        vB = symbolic("BODY_FRAME_LINEAR_VELOCITY", KOOPMAN_CONFIG, self._order)
        gB = symbolic("BODY_FRAME_GRAVITY", KOOPMAN_CONFIG, self._order)
        wB = symbolic("BODY_FRAME_ANGULAR_VELOCITY", KOOPMAN_CONFIG, self._order)
        x = cs.vertcat(pB, vB, gB, wB)

        m = symbolic("m", PARAMETER_CONFIG)
        a = symbolic("a", PARAMETER_CONFIG)
        Ixx = symbolic("Ixx", PARAMETER_CONFIG)
        Iyy = symbolic("Iyy", PARAMETER_CONFIG)
        Izz = symbolic("Izz", PARAMETER_CONFIG)
        k = symbolic("k", PARAMETER_CONFIG)
        c = symbolic("c", PARAMETER_CONFIG)
        r = symbolic("r", PARAMETER_CONFIG)
        s = symbolic("s", PARAMETER_CONFIG)
        theta = cs.SX(cs.vertcat(m, a, Ixx, Iyy, Izz, k, c, r, s))

        J = cs.SX(cs.diag(cs.vertcat(Ixx, Iyy, Izz)))

        # Derive Koopman observables
        ws_N = attitude.get_ws(wB_init, J, self._order)
        Hs_N = attitude.get_Hs(ws_N, J)

        ps = position.get_ps(pB_init, ws_N)
        Ps = position.get_Ps(ps, ws_N, Hs_N, J)

        vs = velocity.get_vs(vB_init, ws_N)
        Os = velocity.get_Os(vs)
        Vs = velocity.get_Vs(vs, ws_N, Hs_N, J)

        gs = gravity.get_gs(gB_init, ws_N)
        Gs = gravity.get_Gs(gs, ws_N, Hs_N, J)

        ws = attitude.get_ws(wB_init, J, self._order)
        Hs = attitude.get_Hs(ws, J)

        # Control input terms
        u = symbolic("MOTOR_SPEED_SQUARED", INPUT_CONFIG)
        K = cs.SX(cs.vertcat(
            cs.SX.zeros(2, self.nu),
            k.T,
        ))
        C = cs.SX(cs.vertcat(
            (k * s).T,
            -(k * r).T,
            cs.SX(alternating_ones(self.nu)).T * c.T,
        ))

        # Process noise
        w = cs.SX.sym("w", self.nw)

        # Continuous-time dynamics
        B = cs.SX( get_input_matrix(Ps, Os, Vs, Gs, Hs, J, m) @ cs.vertcat(K, C) )
        A = cs.SX( get_state_matrix(self._order, self._order) )
        xdot = A @ x + B @ u + w

        # Define dynamics function
        self._f = cs.Function(
            "f_KoopmanLiftedQuadrotorModel",
            [x, u, w, theta, x_init], [xdot]
        )


#TODO: add more models
def CrazyflieModel(a=np.zeros(3)) -> NonlinearQuadrotorModel:
    """
    crazyflie system identification:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143
    """
    cf_params = ModelParameters(
        m=0.027,
        a=a,
        Ixx=1.6571710 * 10**-5,
        Iyy=1.6655602 * 10**-5,
        Izz=2.9261652 * 10**-5,
        k=np.array((1.0, 1.0, 1.0, 1.0)),
        c=np.array((0.005964552, 0.005964552, 0.005964552, 0.005964552)),
        r=np.array((0.0283, 0.0283, -0.0283, -0.0283)),
        s=np.array((0.0283, -0.0283, -0.0283, 0.0283)),
    )
    return NonlinearQuadrotorModel(cf_params)
