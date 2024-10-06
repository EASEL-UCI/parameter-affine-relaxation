#!/usr/bin/python3

import time
import numpy as np
from par.models import CrazyflieModel, ParameterAffineQuadrotorModel
from par.quat import random_unit_quat
from par.mpc import NMPC

nl_model = CrazyflieModel()
aff_model = ParameterAffineQuadrotorModel(nl_model.parameters)

Q = np.diag(np.hstack((
    10.0 * np.ones(3), 5.0 * np.ones(4), 1.0 * np.ones(6)
)))
R = 0.01 * np.eye(4)
Qf = 2.0 * Q
nl_nmpc = NMPC(dt=0.01, N=100, Q=Q, R=R, Qf=Qf, model=nl_model)
aff_nmpc = NMPC(dt=0.01, N=100, Q=Q, R=R, Qf=Qf, model=aff_model)

pos0 = np.random.uniform(low=-2.0, high=2.0, size=3)
att0 = random_unit_quat()
vel0 = np.random.uniform(low=-2.0, high=2.0, size=3)
angvel0 = np.random.uniform(low=-2.0, high=2.0, size=3)
x0 = np.hstack((pos0, att0, vel0, angvel0))

lbu = np.zeros(4)
ubu = 0.15 * np.ones(4)

st = time.time()
nl_nmpc.solve(x=x0, lbu=lbu, ubu=ubu)
et = time.time()
print(f"computation time: {et-st}")
nl_nmpc.plot_trajectory()

st = time.time()
aff_nmpc.solve(x=x0, lbu=lbu, ubu=ubu)
et = time.time()
print(f"computation time: {et-st}")
aff_nmpc.plot_trajectory()


xk = aff_nmpc.get_state_trajectory()
uk = aff_nmpc.get_input_trajectory()
xk_nl = []
for k in range(len(uk)):
    xk_nl += [ nl_model.F(dt=0.01, x=xk[k], u=uk[k]) ]
xk_nl = np.array(xk_nl)
aff_nmpc.plot_trajectory(xk=xk_nl, uk=uk)

