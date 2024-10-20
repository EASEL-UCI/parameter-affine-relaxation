#!/usr/bin/python3

import numpy as np
from numpy.random import uniform
from par.dynamics.vectors import State, Input, ProcessNoise, ModelParameters, \
                                    AffineModelParameters, VectorList
from par.dynamics.models import CrazyflieModel, ParameterAffineQuadrotorModel
from par.utils.math import random_unit_quaternion
from par.mpc import NMPC
from par.mhe import MHPE


# Perturb model parameters
model_nominal = CrazyflieModel(0.1 * np.ones(3))
param_nominal = model_nominal.parameters
perturb = np.random.uniform(low=0.5, high=1.5, size=model_nominal.ntheta)
param_perturb = ModelParameters(perturb * param_nominal.as_array())

# Get inacurrate and accurate models
inacc_model = ParameterAffineQuadrotorModel(
    param_nominal.as_affine(), model_nominal.lbu, model_nominal.ubu)
acc_model = ParameterAffineQuadrotorModel(
    param_perturb.as_affine(), model_nominal.lbu, model_nominal.ubu)

# Init MHPE
dt = 0.05
M = 10
P = np.eye(inacc_model.ntheta)
S = np.eye(inacc_model.nw)
mhpe = MHPE(dt=dt, M=M, P=P, S=S, model=inacc_model, is_verbose=False)

# Init MPC
N = 10
Q = np.eye(inacc_model.nx)
R = 0.01 * np.eye(inacc_model.nu)
Qf = 2.0 * Q
nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=inacc_model)

# Init state
x = State()
x.set_member("POSITION", uniform(-10.0, 10.0, size=3))
x.set_member("ATTITUDE", random_unit_quaternion())
x.set_member("BODY_FRAME_LINEAR_VELOCITY", uniform(-10.0, 10.0, size=3))
x.set_member("BODY_FRAME_ANGULAR_VELOCITY", uniform(-10.0, 10.0, size=3))

# MHE stuff
mhpe.reset_measurements(x)
lb_theta = 0.5 * param_nominal.as_array() * np.ones(inacc_model.ntheta)
ub_theta = 1.5 * param_nominal.as_array() * np.ones(inacc_model.ntheta)
lb_theta_aff_init = ModelParameters(lb_theta).as_affine().as_array()
ub_theta_aff_init = ModelParameters(ub_theta).as_affine().as_array()
lb_theta_aff = AffineModelParameters(np.minimum(lb_theta_aff_init, ub_theta_aff_init))
ub_theta_aff = AffineModelParameters(np.maximum(lb_theta_aff_init, ub_theta_aff_init))

# MPC args
theta = param_perturb.as_affine()
xref = VectorList( N * [State()] )
uref = VectorList( N * [Input()] )
xs_guess = None
us_guess = None

# Sim stuff
w = ProcessNoise()
xs = VectorList()
us = VectorList()

# Iterate sim
sim_len = 100
for k in range(sim_len):
    # Solve, update warmstarts, and get the control input
    nmpc.solve(
        x=x, xref=xref, uref=uref, theta=theta,
        lbu=inacc_model.lbu, ubu=inacc_model.ubu,
        xs_guess=xs_guess, us_guess=us_guess
    )
    xs_guess = nmpc.get_predicted_states()
    us_guess = nmpc.get_predicted_inputs()
    u = us_guess.get(0)

    # Generate Guassian noise on the acceleration
    lin_acc_noise = np.random.normal(loc=0.0, scale=1.0, size=3)
    ang_acc_noise = np.random.normal(loc=0.0, scale=1.0, size=3)
    w.set_member("BODY_FRAME_LINEAR_ACCELERATION", lin_acc_noise)
    w.set_member("BODY_FRAME_ANGULAR_ACCELERATION", ang_acc_noise)

    # Update current state and trajectory history
    x = acc_model.step_sim(dt=dt, x=x, u=u, w=w)
    xs.append(x)
    us.append(u)

    # Get parameter estimate
    mhpe.solve(x, u, lb_theta=lb_theta_aff, ub_theta=ub_theta_aff)
    theta = mhpe.get_parameter_estimate()

    print(f"\ninput {k}: \n{u.as_array()}")
    print(f"\n\n\nstate {k+1}: \n{x.as_array()}")
    print(f"\naffine parameter estimate {k+1}: \n{theta.as_array()}")
    #print(f"\ntrue affine parameter: \n{acc_model.parameters.as_array()}")

normalized_errors = np.zeros(acc_model.ntheta)
for i in range(acc_model.ntheta):
    theta_acc = acc_model.parameters.as_array()[i]
    normalized_errors[i] = ( theta.as_array()[i] - theta_acc ) / theta_acc
print(normalized_errors)
print(f"\nFinal normalized parameter estimate error: {np.linalg.norm(normalized_errors)}")
nmpc.plot_trajectory(xs=xs, us=us, dt=dt, N=sim_len)