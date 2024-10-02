#!/usr/bin/python3

import time
import numpy as np
from par.models import CrazyflieModel
from par.quat import random_unit_quat
from par.mpc import NMPC

nl_model = CrazyflieModel()

Q = np.diag(np.hstack((
    10.0 * np.ones(3), 5.0 * np.ones(4), 1.0 * np.ones(6)
)))
R = 0.1 * np.eye(4)
Qf = 10 * Q
nmpc = NMPC(dt=0.1, N=30, Q=Q, R=R, Qf=Qf, model=nl_model)

pos0 = np.random.uniform(low=-2.0, high=2.0, size=3)
att0 = random_unit_quat()
vel0 = np.random.uniform(low=-2.0, high=2.0, size=3)
angvel0 = np.random.uniform(low=-2.0, high=2.0, size=3)
x0 = np.hstack((pos0, att0, vel0, angvel0))

lbu = np.zeros(4)
ubu = 0.15 * np.ones(4)

st = time.time()
nmpc.solve(x=x0, lbu=lbu, ubu=ubu)
et = time.time()

print(f"computation time: {et-st}")
xk = nmpc.get_state_trajectory()
nmpc.plot_trajectory()

