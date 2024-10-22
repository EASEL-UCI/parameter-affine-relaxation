#!/usr/bin/python3

import numpy as np
from numpy.random import uniform
from par.dynamics.vectors import State, Input, VectorList
from par.dynamics.models import CrazyflieModel, ParameterAffineQuadrotorModel
from par.utils.math import random_unit_quaternion
from par.optimization import NMPC


dt = 0.1
N = 50
model_nl = CrazyflieModel()
theta_aff = model_nl.parameters.as_affine()
model = ParameterAffineQuadrotorModel(
    theta_aff, model_nl.r, model_nl.s, model_nl.lbu, model_nl.ubu)
Q = np.eye(model.nx)
R = np.eye(model.nu)
Qf = 2.0 * Q
nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=model, is_verbose=True)


x = State()
x.set_member('POSITION', uniform(-10.0, 10.0, size=3))
x.set_member('ATTITUDE', random_unit_quaternion())
x.set_member('BODY_FRAME_LINEAR_VELOCITY', uniform(-10.0, 10.0, size=3))
x.set_member('BODY_FRAME_ANGULAR_VELOCITY', uniform(-10.0, 10.0, size=3))

lbu = Input(np.zeros(4))
ubu = Input(0.15 * np.ones(4))

xref = VectorList( N * [State()] )
uref = VectorList( N * [Input()] )


nmpc.solve(x=x, xref=xref, uref=uref, lbu=lbu, ubu=ubu)
nmpc.plot_trajectory()
