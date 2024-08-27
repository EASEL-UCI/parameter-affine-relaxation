#!/usr/bin/python3

import casadi as cs
import numpy as np

from dimpc.config import *
from dimpc.util import symbolic, get_start_index, get_stop_index


class ModelParameters():
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            assert key in kwargs.keys()
            assert type(value) == PARAMETER_CONFIG[key]["type"]
            try:
                if len(value > 1):
                    for member in value:
                        assert type(member) == PARAMETER_CONFIG[key]["member_type"]
            except TypeError: pass

            setattr(self, key, value)


class StateSpaceModel():
    def __init__(
        self,
        parameters: ModelParameters
    ) -> None:
        self._parameters = parameters
        self._xdot = None
        self._x = None
        self._u = None

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
        p = symbolic(STATE_CONFIG, "DRONE_POSITION")
        q = symbolic(STATE_CONFIG, "DRONE_ORIENTATION")
        v = symbolic(STATE_CONFIG, "DRONE_LINEAR_VELOCITY")
        wB = symbolic(STATE_CONFIG, "DRONE_ANGULAR_VELOCITY")
        self._x = cs.SX(cs.vertcat(p, q, v, wB))

        # rotation matrix
        R = cs.SX(cs.vertcat(
            cs.horzcat( 1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2]) ),
            cs.horzcat( 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1]) ),
            cs.horzcat( 2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2) ),
        ))

        # drag terms
        A = cs.SX(cs.diag([param.Ax, param.Ay, param.Az]))

        # Diagonal of inertial matrix
        J = cs.SX(cs.diag([param.Ixx, param.Iyy, param.Izz]))

        # control allocation matrix
        B = cs.SX(cs.vertcat(
            param.kf * cs.SX(param.yB).reshape((1, len(param.yB))),
            param.kf * cs.SX(param.xB).reshape((1, len(param.xB))),
            param.km * cs.horzcat(-1, 1, -1, 1),
        ))

        # gravity vector
        g = cs.SX(cs.vertcat(0, 0, -GRAVITY))

        # thrust of motors
        self._u = symbolic(INPUT_CONFIG, "DRONE_THRUSTS")
        T = cs.SX(cs.vertcat(
            0, 0, param.kf * (self._u[0] + self._u[1] + self._u[2] + self._u[3])
        ))

        # double integrator dynamics
        self._xdot = cs.SX(cs.vertcat(
            v,
            -cs.dot(q[1:], 0.5*wB),
            0.5 * q[0] * wB + cs.cross(q[1:], wB),
            g + (R @ T - A @ v) / param.m,
            cs.inv(J) @ (B @ self._u - cs.cross(wB, J @ wB))
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
        A = cs.jacobian(self._xdot, self._x)
        A = cs.substitute(A, self._x, xref)

        B = cs.jacobian(self._xdot, self._u)
        B = cs.substitute(B, self._u, uref)

        self._xdot = A @ self._x + B @ self._u


class DeliveryQuadrotorModel(NonlinearQuadrotorModel):
    def __init__(
        self,
        drone_parameters: ModelParameters,
        payload_parameters: ModelParameters,
    ) -> None:
        super().__init__(drone_parameters)
        self._add_payload_model(payload_parameters)

    def _add_payload_model(self, param: ModelParameters) -> None:
        #TODO: INCLUDE PAYLOAD INTO ORIGINAL PARAMETERS
        p = symbolic(STATE_CONFIG, "PAYLOAD_POSITION_0")
        v = symbolic(STATE_CONFIG, "PAYLOAD_VELOCITY_0")
        u = symbolic(INPUT_CONFIG, "PAYLOAD_RELEASE")

        acc = self._xdot[
            get_start_index(STATE_CONFIG, "DRONE_LINEAR_VELOCITY") :
            get_stop_index(STATE_CONFIG, "DRONE_LINEAR_VELOCITY")
        ]

        g = cs.SX(cs.vertcat(0.0, 0.0, -GRAVITY))

        sig = 1 / (1 + cs.exp(CONTACT_ACTIVATION_FACTOR * p[-1]))
        ground = cs.SX(cs.vertcat(
            -CONTACT_DAMPER_FACTOR * v[0],
            -CONTACT_DAMPER_FACTOR * v[1],
            -(CONTACT_SPRING_FACTOR * p[2] + CONTACT_DAMPER_FACTOR * v[2])
        ))

        self._xdot = cs.SX(cs.vertcat(
            self._xdot,
            v,
            (1-u) * acc + u * (g + sig*ground/param.m)
        ))
        self._x = cs.SX(cs.vertcat(self._x, p, v))
        self._u = cs.SX(cs.vertcat(self._u, u))


def CrazyflieModel(
    Ax=0.0,
    Ay=0.0,
    Az=0.0,
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
        xB=[0.0283, 0.0283, -0.0283, -0.0283],
        yB=[0.0283, -0.0283, -0.0283, 0.0283],
    )
    return NonlinearQuadrotorModel(cf_params)
