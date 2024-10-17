#!/usr/bin/python3

import numpy as np
from numpy.random import uniform
from par.dynamics.vectors import State, Input, DynamicsVectorList
from par.dynamics.models import CrazyflieModel
from par.utils.math import random_unit_quaternion
from par.mpc import NMPC


dt = 0.1
N = 20
Q = np.diag(np.hstack(( 10.0 * np.ones(3), 5.0 * np.ones(4), 1.0 * np.ones(6) )))
R = 0.01 * np.eye(4)
Qf = 2.0 * Q
nl_model = CrazyflieModel()
nl_nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=nl_model, is_verbose=False)


x = State()
x.set_member("POSITION", uniform(-10.0, 10.0, size=3))
x.set_member("ATTITUDE", random_unit_quaternion())
x.set_member("BODY_FRAME_LINEAR_VELOCITY", uniform(-10.0, 10.0, size=3))
x.set_member("BODY_FRAME_ANGULAR_VELOCITY", uniform(-10.0, 10.0, size=3))

lbu = Input(np.zeros(4))
ubu = Input(0.15 * np.ones(4))

xref = DynamicsVectorList( N * [State()] )
uref = DynamicsVectorList( N * [Input()] )

xs_guess = None
us_guess = None

xs = DynamicsVectorList()
us = DynamicsVectorList()


sim_length = 100
for k in range(sim_length):
    # Solve, update warmstarts, and get the control input
    nl_nmpc.solve(
        x=x, xref=xref, uref=uref, lbu=lbu, ubu=ubu,
        xs_guess=xs_guess, us_guess=us_guess)
    xs_guess = nl_nmpc.get_predicted_states()
    us_guess = nl_nmpc.get_predicted_inputs()
    u = us_guess.get(0)

    # Generate Guassian noise on the second order terms
    second_order_noise = np.random.normal(loc=1.0, scale=1.0, size=6)
    w = np.hstack((np.zeros(7), second_order_noise))

    # Update current state and trajectory history
    x = State(nl_model.F(dt=dt, x=x.as_array(), u=u.as_array(), w=w))
    xs.append(x)
    us.append(u)

    print(f"\ninput {k}: \n{u.as_array()}")
    print(f"\n\n\nstate {k+1}: \n{x.as_array()}")

nl_nmpc.plot_trajectory(xs=xs, us=us, dt=dt, N=sim_length)
