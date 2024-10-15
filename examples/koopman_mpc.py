#!/usr/bin/python3

import time
import numpy as np
from par.models import CrazyflieModel, KoopmanLiftedQuadrotorModel
from par.mpc import NMPC
from par.constants import GRAVITY

order = 4
km_init = np.hstack((np.zeros(9), -GRAVITY, 0.0, 0.0))
nl_model = CrazyflieModel()
km_model = KoopmanLiftedQuadrotorModel(
    initial_state=km_init, observables_order=4, parameters=nl_model.parameters)

dt = 0.1
N = 100
Q = np.diag(np.hstack((
    10.0 * np.ones(3*order), 1.0 * np.ones(3*order), 0.0 * np.ones(3*order),
    5.0 * np.ones(3*order)
)))
R = 0.01 * np.eye(4)
Qf = 2.0 * Q
km_mpc = NMPC(dt=dt, N=N, Q=Q, R=R, Qf=Qf, model=km_model)
