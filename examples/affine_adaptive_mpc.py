#!/usr/bin/python3

import numpy as np
from par.dynamics.vectors import State, Input, ProcessNoise, ModelParameters, \
                                    AffineModelParameters, VectorList
from par.dynamics.models import CrazyflieModel, ParameterAffineQuadrotorModel
from par.utils.math import random_unit_quaternion
from par.optimization import NMPC, MHPE


# Perturb model parameters
model_nominal = CrazyflieModel(0.1 * np.ones(3))
param_nominal = model_nominal.parameters
perturb = np.random.uniform(low=0.5, high=1.5, size=model_nominal.ntheta)
param_perturb = ModelParameters(perturb * param_nominal.as_array())

# Get inacurrate and accurate models
model_inacc = ParameterAffineQuadrotorModel(
    param_nominal.as_affine(),
    model_nominal.r, model_nominal.s,
    model_nominal.lbu, model_nominal.ubu
)
model_acc = ParameterAffineQuadrotorModel(
    param_perturb.as_affine(),
    model_nominal.r, model_nominal.s,
    model_nominal.lbu, model_nominal.ubu
)

# Init MHPE
dt = 0.05
M = 10
P = np.diag(np.hstack((
    1.0, 1.0 * np.ones(3), 1e-5 * np.ones(3), 1.0 * np.ones(3)
)))
S = np.eye(model_inacc.nw)
mhpe = MHPE(dt=dt, M=M, P=P, S=S, model=model_inacc, plugin='proxqp')

# Init MPC
N = 20
Q = np.eye(model_inacc.nx)
R = 0.0 * np.eye(model_inacc.nu)
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
lb_theta = 0.5 * param_nominal.as_array() * np.ones(model_nominal.ntheta)
ub_theta = 1.5 * param_nominal.as_array() * np.ones(model_nominal.ntheta)
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
    lin_acc_noise = np.random.uniform(low=-10.0, high=10.0, size=3)
    ang_acc_noise = np.random.uniform(low=-10.0, high=10.0, size=3)
    w.set_member('BODY_FRAME_LINEAR_ACCELERATION', lin_acc_noise)
    w.set_member('BODY_FRAME_ANGULAR_ACCELERATION', ang_acc_noise)

    # Update current state and trajectory history
    x = model_acc.step_sim(dt=dt, x=x, u=u, w=w)
    xs.append(x)
    us.append(u)

    # Get parameter estimate
    mhpe.solve(x, u, lb_theta=lb_theta_aff, ub_theta=ub_theta_aff)
    theta = mhpe.get_parameter_estimate()

    print(f'\ninput {k}: \n{u.as_array()}')
    print(f'\n\n\nstate {k+1}: \n{x.as_array()}')
    print(f'\naffine parameter estimate {k+1}: \n{theta.as_array()}')
print(f'\nnominal affine parameter: \n{model_inacc.parameters.as_array()}')
print(f'\ntrue affine parameter: \n{model_acc.parameters.as_array()}')

normalized_errors = np.zeros(model_acc.ntheta)
for i in range(model_acc.ntheta):
    theta_acc = model_acc.parameters.as_array()[i]
    normalized_errors[i] = ( theta.as_array()[i] - theta_acc ) / theta_acc

print(f'\nFinal parameter estimate error: {
    np.linalg.norm(normalized_errors)
}')

nmpc.plot_trajectory(xs=xs, us=us, dt=dt, N=sim_len)
