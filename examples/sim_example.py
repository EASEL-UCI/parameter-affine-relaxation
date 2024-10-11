#!/usr/bin/python3

import time
import numpy as np
from par.models import CrazyflieModel
from par.quat import random_unit_quat
from par.mpc import NMPC
from par.mhe import MHPE


dt = 0.1
N = 10
Q = np.diag(np.hstack((
    10.0 * np.ones(3), 5.0 * np.ones(4), 1.0 * np.ones(6)
)))
R = 0.01 * np.eye(4)
Qf = 2.0 * Q
nl_model = CrazyflieModel()
nl_nmpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=nl_model)


'''
M = 5
P = np.eye(23)
S = np.eye(13)
mhpe = MHPE(dt=dt, M=M, P=P, S=S, model=nl_model)
'''


lbu = np.zeros(4)
ubu = 0.15 * np.ones(4)


pos0 = np.random.uniform(low=-10.0, high=10.0, size=3)
att0 = random_unit_quat()
vel0 = np.random.uniform(low=-10.0, high=10.0, size=3)
angvel0 = np.random.uniform(low=-10.0, high=10.0, size=3)
x = np.hstack((pos0, att0, vel0, angvel0))
xk_guess = None
uk_guess = None
xk = [x]
uk = []


sim_length = 50
for k in range(sim_length):
    # Solve, update warmstarts, and get the control input
    nl_nmpc.solve(x=x, lbu=lbu, ubu=ubu, xk_guess=xk_guess, uk_guess=uk_guess)
    xk_guess = nl_nmpc.get_state_trajectory()
    uk_guess = nl_nmpc.get_input_trajectory()
    u = uk_guess[0, :]
    second_order_noise = np.random.normal(loc=1.0, scale=1.0, size=6)
    w = np.hstack((np.zeros(7), second_order_noise))

    # Update current state and trajectory history
    x = nl_model.F(dt=dt, x=x, u=u, w=w)
    xk += [x]
    uk += [u]


xk = np.array(xk)
uk = np.array(uk)
nl_nmpc.plot_trajectory(xk=xk, uk=uk, dt=dt, N=sim_length)
