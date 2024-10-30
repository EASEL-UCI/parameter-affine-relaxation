#!/usr/bin/env python3

import numpy as np
from numpy.random import uniform
from par.dynamics.vectors import State, Input, ProcessNoise, VectorList
from par.dynamics.models import CrazyflieModel
from par.utils.math import random_unit_quaternion
from par.optimization import NMPC


dt = 0.05
N = 20
model = CrazyflieModel()
Q_diag = State()
Q_diag.set_member('position_wf', 10.0 * np.ones(3))
Q_diag.set_member('attitude', 1.0 * np.ones(4))
Q_diag.set_member('linear_velocity_bf', np.ones(3))
Q_diag.set_member('angular_velocity_bf', 10.0 * np.ones(3))
Q = np.diag(Q_diag.as_array())
R = 0.001 * np.eye(model.nu)
Qf = 2.0 * Q
nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=model, is_verbose=False)


x = State()
x.set_member('position_wf', uniform(-10.0, 10.0, size=3))
x.set_member('attitude', random_unit_quaternion())
x.set_member('linear_velocity_bf', uniform(-10.0, 10.0, size=3))
x.set_member('angular_velocity_bf', uniform(-10.0, 10.0, size=3))

w = ProcessNoise()

xref = VectorList( N * [State()] )
uref = VectorList( N * [Input()] )

xs_guess = None
us_guess = None

xs = VectorList()
us = VectorList()


sim_length = 200
for k in range(sim_length):
    # Solve, update warmstarts, and get the control input
    nmpc.solve(
        x=x, xref=xref, uref=uref, lbu=model.lbu, ubu=model.ubu,
        xs_guess=xs_guess, us_guess=us_guess
    )
    xs_guess = nmpc.get_predicted_states()
    us_guess = nmpc.get_predicted_inputs()
    u = us_guess.get(0)

    # Generate Guassian noise on the second order terms
    lin_acc_noise = np.random.normal(loc=1.0, scale=1.0, size=3)
    ang_acc_noise = np.random.normal(loc=1.0, scale=1.0, size=3)
    w.set_member('linear_acceleration_bf', lin_acc_noise)
    w.set_member('angular_acceleration_bf', ang_acc_noise)

    # Update current state and trajectory history
    x = model.step_sim(dt=dt, x=x, u=u, w=w)
    xs.append(x)
    us.append(u)

    print(f'\ninput {k}: \n{u.as_array()}')
    print(f'\n\n\nstate {k+1}: \n{x.as_array()}')

nmpc.plot_trajectory(xs=xs, us=us, dt=dt, N=sim_length)
