#!/usr/bin/python3

import time
import numpy as np
from par.models import CrazyflieModel, KoopmanLiftedQuadrotorModel
from par.mpc import NMPC

nl_model = CrazyflieModel()
km_model = KoopmanLiftedQuadrotorModel(
    initial_state=np.hstack((np.zeros(9), -9.81, 0.0, 0.0)),
    observables_order=4, parameters=nl_model.parameters
)
