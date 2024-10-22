#!/usr/bin/python3

import pickle

import numpy as np

from par.dynamics.vectors import State, Input, ProcessNoise, ModelParameters, \
                                    VectorList
from par.dynamics.models import CrazyflieModel, NonlinearQuadrotorModel
from par.utils.math import random_unit_quaternion
from par.optimization import NMPC, MHPE
from par.utils.data import *


# Perturb model parameters
model_inacc = CrazyflieModel(0.1 * np.ones(3))
param_nominal = model_inacc.parameters
perturb = np.random.uniform(low=0.5, high=1.5, size=model_inacc.ntheta)
param_perturb = ModelParameters(perturb * param_nominal.as_array())
model_acc = NonlinearQuadrotorModel(
    param_perturb, model_inacc.r, model_inacc.s, model_inacc.lbu, model_inacc.ubu)

# Init MHPE
dt = 0.05
M = 10
P = np.eye(model_inacc.ntheta)
S = np.eye(model_inacc.nw)
mhpe = MHPE(dt=dt, M=M, P=P, S=S, model=model_inacc, plugin='ipopt')

# Init MPC
N = 20
Q = np.eye(model_inacc.nx)
R = 0.01 * np.eye(model_inacc.nu)
Qf = 2.0 * Q
nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=model_inacc)

# Init state
x = State()
x.set_member('POSITION', np.random.uniform(-10.0, 10.0, size=3))
x.set_member('ATTITUDE', random_unit_quaternion())
x.set_member('BODY_FRAME_LINEAR_VELOCITY', np.random.uniform(-10.0, 10.0, size=3))
x.set_member('BODY_FRAME_ANGULAR_VELOCITY', np.random.uniform(-10.0, 10.0, size=3))

# MHE stuff
mhpe.reset_measurements(x)
lb_theta = ModelParameters(
    0.5 * param_nominal.as_array() * np.ones(model_inacc.ntheta))
ub_theta = ModelParameters(
    1.5 * param_nominal.as_array() * np.ones(model_inacc.ntheta))
print(lb_theta.as_array())
print(ub_theta.as_array())

# MPC args
theta = model_inacc.parameters
xref = VectorList( N * [State()] )
uref = VectorList( N * [Input()] )
xs_guess = None
us_guess = None

# Sim stuff
w = ProcessNoise()
xs = VectorList()
us = VectorList()

# Data collection
data = []

# Iterate sim
sim_len = 200
for k in range(sim_len):
    # Solve, update warmstarts, and get the control input
    nmpc.solve(
        x=x, xref=xref, uref=uref, theta=theta,
        lbu=model_inacc.lbu, ubu=model_inacc.ubu,
        xs_guess=xs_guess, us_guess=us_guess
    )
    xs_guess = nmpc.get_predicted_states()
    us_guess = nmpc.get_predicted_inputs()
    u = us_guess.get(0)

    # Generate Guassian noise on the acceleration
    lin_acc_noise = np.random.normal(loc=1.0, scale=1.0, size=3)
    ang_acc_noise = np.random.normal(loc=1.0, scale=1.0, size=3)
    w.set_member('BODY_FRAME_LINEAR_ACCELERATION', lin_acc_noise)
    w.set_member('BODY_FRAME_ANGULAR_ACCELERATION', ang_acc_noise)

    # Update current state and trajectory history
    x = model_acc.step_sim(dt=dt, x=x, u=u, w=w)
    xs.append(x)
    us.append(u)

    # Get parameter estimate
    mhpe.solve(x, u)
    theta = mhpe.get_parameter_estimate()

    # Data collection
    if k >= M:
        data_k = SimData(
            x, u, w, theta, xref.get(0), uref.get(0), param_perturb, Q, R,
            mhpe.get_full_solution(), mhpe.get_solver_stats()
        )
        data += [data_k]

    print(f'\ninput {k}: \n{u.as_array()}')
    print(f'\n\n\nstate {k+1}: \n{x.as_array()}')
    print(f'\nparameter estimate {k+1}: \n{theta.as_array()}')
print(f'\nnominal affine parameter: \n{model_inacc.parameters.as_array()}')
print(f'\ntrue affine parameter: \n{model_acc.parameters.as_array()}')

print(get_mhpe_non_convergences(data))
print(get_average_mhpe_solve_time(data))
print(get_cost(data))

nmpc.plot_trajectory(xs=xs, us=us, dt=dt, N=sim_len)
