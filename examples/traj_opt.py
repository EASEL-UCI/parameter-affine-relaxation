#!/usr/bin/python3

import numpy as np
from numpy.random import uniform
from par.dynamics.vectors import State, Input, VectorList
from par.dynamics.models import CrazyflieModel
from par.utils.math import random_unit_quaternion
from par.mpc import NMPC


dt = 0.1
N = 50
Q = np.diag(np.hstack(( 2.0 * np.ones(3), 2.0 * np.ones(4), 1.0 * np.ones(6) )))
R = 0.01 * np.eye(4)
Qf = 2.0 * Q
nl_model = CrazyflieModel()
nl_nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=nl_model, is_verbose=True)


x = State()
x.set_member("POSITION", uniform(-10.0, 10.0, size=3))
x.set_member("ATTITUDE", random_unit_quaternion())
x.set_member("BODY_FRAME_LINEAR_VELOCITY", uniform(-10.0, 10.0, size=3))
x.set_member("BODY_FRAME_ANGULAR_VELOCITY", uniform(-10.0, 10.0, size=3))

lbu = Input(np.zeros(4))
ubu = Input(0.15 * np.ones(4))

xref = VectorList( N * [State()] )
uref = VectorList( N * [Input()] )


nl_nmpc.solve(x=x, xref=xref, uref=uref, lbu=lbu, ubu=ubu)
nl_nmpc.plot_trajectory()
