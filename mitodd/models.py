#!/usr/bin/python3

import casadi as cs
import numpy as np


class QuadrotorModel():
    def __init__(
        self,
        parameters: dict,
        name="QuadrotorModel",
    ) -> None:
        self.name = name
        config = self._parameter_config()
        for key, value in parameters:
            assert key in config.keys()
            assert type(value) == config[key]
            setattr(self, key, value)
        self._model = self._get_model()

    @property
    def xdot(self):
        return self._model["xdot"]

    @property
    def x(self):
        return self._model["x"]
    
    @property
    def u(self):
        return self._model["u"]

    def _parameter_config():
        return {
            "m": float, "Ixx": float, "Iyy": float, "Izz": float,
            "Ax": float, "Ay": float, "Az": float, "xB": np.ndarray, "yB": np.ndarray,
            "kf": float, "km": float, "umin": float, "umax": float, "name": str,
        }

    def _get_model(self) -> dict:
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
            x, y, z, q0, q1, q2, q3,
            xdot, ydot, zdot, p, q, r
        ))

        # rotation matrix
        R = cs.SX(cs.vertcat(
            cs.horzcat( 1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2) ),
            cs.horzcat( 2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1) ),
            cs.horzcat( 2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2) ),
        ))

        # drag terms
        A = cs.SX(np.diag([self.Ax, self.Ay, self.Az]))

        # Diagonal of inertial matrix
        J = cs.SX(np.diag([self.Ixx, self.Iyy, self.Izz]))

        # control allocation matrix
        B = cs.SX(cs.vertcat(
            self.kf * self.yB.reshape(1, self.yB.shape[0]),
            self.kf * -self.xB.reshape(1, self.xB.shape[0]),
            self.km * cs.horzcat(-1, 1, -1, 1),
        ))

        # gravity vector
        g = cs.SX(cs.vertcat(0, 0, -9.81))

        # thrust of motors 1 to 4
        U = cs.SX.sym("u", 4)
        T = cs.SX(cs.vertcat(
            0, 0, self.kf * (self.u[0]+self.u[1]+self.u[2]+self.u[3])
        ))

        # double integrator dynamics
        qv = cs.SX(cs.vertcat(q1,q2,q3))
        v = cs.SX(cs.vertcat(xdot, ydot, zdot))
        wB = cs.SX(cs.vertcat(p, q, r))

        Xdot = cs.SX(cs.vertcat(
            v,
            -cs.dot(qv, 0.5*wB),
            0.5 * q0 * wB + cs.cross(qv, wB),
            (R @ T - A @ v) / self.m + g,
            cs.inv(J) @ (B @ self.u - cs.cross(wB, J @ wB))
        ))

        return {"xdot": Xdot, "x": X, "u": U}


class LinearizedQuadrotor(QuadrotorModel):
    def __init__(
        self,
        parameters: dict,
        xref: cs.SX,
        uref: cs.SX,
        t: cs.SX,
        name="LinearizedQuadrotorModel"
    ) -> None:
        super().__init__(parameters, name)
        self._model = self._linearize_model(self.model, xref, uref, t)

    def _linearize_model(
        self,
        model: dict,
        xref: cs.SX,
        uref:cs.SX,
        t: cs.SX,
    ) -> dict:
        A = cs.jacobian(model["xdot"], model["x"])
        A = cs.substitute(A, model["x"], xref)
        
        B = cs.jacobian(model["xdot"], model["u"])
        B = cs.substitute(B, model["u"], uref)
        
        model["xdot"] = A @ model["x"] + B @ model["u"]
        return model

def Crazyflie(
    Ax: float,
    Ay: float,
    Az: float,
) -> QuadrotorModel:
    """
    crazyflie system identification:
    https://www.research-collection.ethz.ch/handle/20.500.11850/214143
    """

    cf_params = {
        "Ax": Ax,
        "Ay": Ay,
        "Az": Az,
        "m": 0.027,
        "Ixx": 1.6571710 * 10**-5,
        "Iyy": 1.6655602 * 10**-5,
        "Izz": 2.9261652 * 10**-5,
        "xB" : 0.0283 * np.array([1, 1, -1, -1]),
        "yB" : 0.0283 * np.array([1, -1, -1, 1]),
        "kf" : 1.0,
        "umax" : 0.15 * np.ones(4),
        "km" : 0.005964552,
        "umin" : np.zeros(4),
    }
    return QuadrotorModel(cf_params)
