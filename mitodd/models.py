#!/usr/bin/python3

from abc import ABC, abstractmethod

import casadi as cs
import numpy as np


PARAMETER_CONFIG  = {
    "m": float,
    "Ixx": float,
    "Iyy": float,
    "Izz": float,
    "Ax": float,
    "Ay": float,
    "Az": float,
    "kf": float,
    "km": float,
    "umin": float,
    "umax": float,
    "xB": np.ndarray,
    "yB": np.ndarray,
}


class ModelParameters():
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            assert key in PARAMETER_CONFIG.keys()
            assert type(value) == PARAMETER_CONFIG[key]
            setattr(self, key, value)


class Model(ABC):
    @property
    @abstractmethod
    def xdot(self) -> cs.SX:
        pass
    
    @property
    @abstractmethod
    def x(self) -> cs.SX:
        pass
    
    @property
    @abstractmethod
    def u(self) -> cs.SX:
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> ModelParameters:
        pass


class QuadrotorModel(Model):
    def __init__(
        self,
        parameters: ModelParameters,
    ) -> None:
        self._set_model(parameters)
        self._parameters = parameters

    @property
    def xdot(self):
        return self._xdot

    @property
    def x(self):
        return self._x

    @property
    def u(self):
        return self._u

    @property
    def parameters(self) -> ModelParameters:
        return self._parameters

    def _set_model(
        self,
        param: ModelParameters
    ) -> None:
        model = self._derive_model(param)
        self._xdot = model["xdot"]
        self._x = model["x"]
        self._u = model["u"]

    def _derive_model(
        self,
        param: ModelParameters
    ) -> dict:
        x = cs.SX.sym("x")
        y = cs.SX.sym("y")
        z = cs.SX.sym("z")
        q0 = cs.SX.sym("q0")
        q1 = cs.SX.sym("q1")
        q2 = cs.SX.sym("q2")
        q3 = cs.SX.sym("q3")
        xdot = cs.SX.sym("xdot")
        ydot = cs.SX.sym("ydot")
        zdot = cs.SX.sym("zdot")
        p = cs.SX.sym("p")
        q = cs.SX.sym("q")
        r = cs.SX.sym("r")

        X = cs.SX(cs.vertcat(
            x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r
        ))

        # rotation matrix
        R = cs.SX(cs.vertcat(
            cs.horzcat( 1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2) ),
            cs.horzcat( 2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1) ),
            cs.horzcat( 2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2) ),
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

        # thrust of motors 1 to 4
        U = cs.SX.sym("u", 4)
        T = cs.SX(cs.vertcat(
            0, 0, param.kf *
                (U[0]+U[1]+U[2]+U[3])
        ))

        # double integrator dynamics
        qv = cs.SX(cs.vertcat(q1,q2,q3))
        v = cs.SX(cs.vertcat(xdot, ydot, zdot))
        wB = cs.SX(cs.vertcat(p, q, r))
        Xdot = cs.SX(cs.vertcat(
            v,
            -cs.dot(qv, 0.5*wB),
            0.5 * q0 * wB + cs.cross(qv, wB),
            (R @ T - A @ v) / param.m + g,
            cs.inv(J) @ (B @ U - cs.cross(wB, J @ wB))
        ))

        return {"xdot": Xdot, "x": X, "u": U}


class LinearizedQuadrotorModel(QuadrotorModel):
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

        self.xdot = A @ self.x + B @ self.u


def Crazyflie(
    Ax: float,
    Ay: float,
    Az: float,
) -> QuadrotorModel:
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

    return QuadrotorModel(cf_params)
