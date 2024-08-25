#!/usr/bin/python3

import casadi as cs
import numpy as np

from mitodd.config import PARAMETER_CONFIG, STATE_VECTOR_CONFIG, INPUT_VECTOR_CONFIG


class ModelParameters():
    def __init__(self, **kwargs) -> None:
        keys = list(kwargs.keys())
        for key, value in kwargs.items():
            if key in keys: keys.remove(key)
            else: raise AssertionError

            assert type(value) == PARAMETER_CONFIG[key]["type"]

            try:
                if len(value > 1):
                    for member in value:
                        assert type(member) == PARAMETER_CONFIG[key]["member_type"]
            except TypeError: pass

            setattr(self, key, value)

        if len(keys):
            raise AssertionError


class StateSpaceModel():
    def __init__(
        self,
        parameters: ModelParameters
    ) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> ModelParameters:
        return self._parameters

    @property
    def xdot(self) -> cs.SX:
        return self._xdot

    @property
    def x(self) -> cs.SX:
        return self._x

    @property
    def u(self) -> cs.SX:
        return self._u


class NonlinearQuadrotorModel(StateSpaceModel):
    def __init__(
        self,
        parameters: ModelParameters,
    ) -> None:
        super().__init__(parameters)
        self._set_model(parameters)

    def _set_model(
        self,
        param: ModelParameters
    ) -> dict:
        p = cs.SX.sym(
            "DRONE_POSITION", STATE_VECTOR_CONFIG["DRONE_POSITION"]["dimensions"]
)
        q = cs.SX.sym(
            "DRONE_ORIENTATION", STATE_VECTOR_CONFIG["DRONE_ORIENTATION"]["dimensions"]
        )
        v = cs.SX.sym(
            "DRONE_LINEAR_VELOCITY", STATE_VECTOR_CONFIG["DRONE_LINEAR_VELOCITY"]["dimensions"]
        )
        wB = cs.SX.sym(
            "DRONE_ANGULAR_VELOCITY", STATE_VECTOR_CONFIG["DRONE_ANGULAR_VELOCITY"]["dimensions"]
        )
        self._x = cs.SX(cs.vertcat(p, q, v, wB))

        # rotation matrix
        R = cs.SX(cs.vertcat(
            cs.horzcat( 1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]) ),
            cs.horzcat( 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1]) ),
            cs.horzcat( 2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2) ),
        ))

        # drag terms
        A = cs.SX(np.diag([param.Ax, param.Ay, param.Az]))

        # Diagonal of inertial matrix
        J = cs.SX(np.diag([param.Ixx, param.Iyy, param.Izz]))

        # control allocation matrix
        B = cs.SX(cs.vertcat(
            param.kf * param.yB.reshape(1, param.yB.shape[0]),
            param.kf * -param.xB.reshape(1, param.xB.shape[0]),
            param.km * cs.horzcat(-1, 1, -1, 1),
        ))

        # gravity vector
        g = cs.SX(cs.vertcat(0, 0, -9.81))

        # thrust of motors
        self._u = cs.SX.sym(
            "DRONE_THRUSTS", INPUT_VECTOR_CONFIG["DRONE_THRUSTS"]["dimensions"]
        )
        T = cs.SX(cs.vertcat(
            0, 0, param.kf * (self.u[0] + self.u[1] + self.u[2] + self.u[3])
        ))

        # double integrator dynamics
        self._xdot = cs.SX(cs.vertcat(
            v,
            -cs.dot(q[1:], 0.5*wB),
            0.5 * q[0] * wB + cs.cross(q[1:], wB),
            (R @ T - A @ v) / param.m + g,
            cs.inv(J) @ (B @ self.u - cs.cross(wB, J @ wB))
        ))


class LinearizedQuadrotorModel(NonlinearQuadrotorModel):
    def __init__(
        self,
        parameters: ModelParameters,
        xref: cs.SX,
        uref: cs.SX,
    ) -> None:
        super().__init__(parameters)
        self._linearize_model(xref, uref)

    def _linearize_model(
        self,
        xref: cs.SX,
        uref:cs.SX,
    ) -> dict:
        A = cs.jacobian(self.xdot, self.x)
        A = cs.substitute(A, self.x, xref)

        B = cs.jacobian(self.xdot, self.u)
        B = cs.substitute(B, self.u, uref)

        self._xdot = A @ self.x + B @ self.u


class DeliveryQuadrotorModel(NonlinearQuadrotorModel):
    def __init__(
        self,
        parameters: ModelParameters,
    ) -> None:
        super().__init__(parameters)
        self._add_payload_to_model()

    def _add_payload_to_model():
        pass


def CrazyflieModel(
    Ax: float,
    Ay: float,
    Az: float,
) -> NonlinearQuadrotorModel:
    """
    crazyflie system identification:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143
    """
    cf_params = ModelParameters(
        Ax=Ax,
        Ay=Ay,
        Az=Az,
        m=0.027,
        Ixx=1.6571710 * 10**-5,
        Iyy=1.6655602 * 10**-5,
        Izz=2.9261652 * 10**-5,
        kf=1.0,
        km=0.005964552,
        umin=0.0,
        umax=0.15,
        xB=0.0283 * np.array([1, 1, -1, -1]),
        yB=0.0283 * np.array([1, -1, -1, 1]),
    )
    return NonlinearQuadrotorModel(cf_params)
