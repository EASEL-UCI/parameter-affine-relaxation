#!/usr/bin/python3

import numpy as np
from par.dynamics.models import CrazyflieModel, KoopmanLiftedQuadrotorModel
from par.mpc import NMPC
from par.constants import GRAVITY
from par.utils.math import random_unit_quaternion


order = 6
nl_model = CrazyflieModel()
km_model = KoopmanLiftedQuadrotorModel(
    observables_order=order, parameters=nl_model.parameters)


dt = 0.1
N = 10
Q = np.diag(np.hstack((
    10.0 * np.ones(3), 10.0 * np.ones(3*(order-1)),   # position
    1.0 * np.ones(3), 1.0 * np.ones(3*(order-1)),    # velocity
    np.zeros(3*order),                                # gravity
    1.0 * np.ones(3), 1.0 * np.ones(3*(order-1)),    # angular velocity
)))
R = 0.01 * np.eye(4)
Qf = 2.0 * Q
km_mpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=km_model, is_verbose=True)


lbu = np.zeros(4)
ubu = 0.15 * np.ones(4)


pos0 = np.random.uniform(low=-1.0, high=1.0, size=3)
#att0 = random_unit_quaternion()
att0 = np.hstack((1.0, np.zeros(3)))
vel0 = np.random.uniform(low=-1.0, high=1.0, size=3)
angvel0 = np.random.uniform(low=-1.0, high=1.0, size=3)
#angvel0 = np.zeros(3)
x = np.hstack((pos0, att0, vel0, angvel0))


xk_guess = None
uk_guess = None
xk = [x]
uk = []


sim_length = 100
for k in range(sim_length):
    # Solve, update warmstarts, and get the control input
    km_mpc.solve(x=x, lbu=lbu, ubu=ubu, xk_guess=xk_guess, uk_guess=uk_guess)
    xk_guess = km_mpc.get_predicted_states()
    uk_guess = km_mpc.get_predicted_inputs()
    u = uk_guess[0, :]

    # Generate Guassian noise on the second order terms
    second_order_noise = np.random.normal(loc=1.0, scale=1.0, size=6)
    w = np.hstack((np.zeros(7), second_order_noise))
    w = np.zeros(13)

    # Update current state and trajectory history
    x = nl_model.F(dt=dt, x=x, u=u, w=w)
    xk += [x]
    uk += [u]
    print(f"\ninput {k}: \n{u}")
    print(f"\n\n\nstate {k+1}: \n{x}")

print(km_model.convert_nominal_to_koopman_lifted_state(xk[-1]))
print(xk_guess[1])

xk = np.array(xk)
uk = np.array(uk)
km_mpc.plot_trajectory(xk=xk, uk=uk, dt=dt, N=sim_length)
