#!/usr/bin/env python3

import numpy as np

from par.dynamics.vectors import ProcessNoise, VectorList, State
from par.utils.math import random_unit_quaternion
from exp_consts import *


def get_process_noise_seed() -> VectorList:
    ws = VectorList()
    for i in range(SIM_LEN):
        w_arr = np.random.uniform(NOISE_MIN, NOISE_MAX)
        ws.append(ProcessNoise(w_arr))
    return ws


def get_random_state() -> State:
    x = State()
    x['position_wf'] = np.random.uniform(POS_MIN, POS_MAX)
    x['attitude'] = random_unit_quaternion()
    x['linear_velocity_bf'] = np.random.uniform(VEL_MIN, VEL_MAX)
    x['angular_velocity_bf'] = np.random.uniform(VEL_MIN, VEL_MAX)
    return x
