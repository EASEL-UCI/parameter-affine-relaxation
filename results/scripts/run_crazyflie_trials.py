#!/usr/bin/env python3

import numpy as np

from par.dynamics.models import CrazyflieModel

from consts.trials import *
from consts.crazyflie import *
from consts.paths import DATA_PATH, CRAZYFLIE_PATH
from run_trials import run_trials


def main():
    a = np.array([0.01, 0.01, 0.05])
    nominal_model = CrazyflieModel(a)
    data_path = DATA_PATH + CRAZYFLIE_PATH
    run_trials(
        nominal_model=nominal_model,
        M=M, P=P, S=S, P_aff=P_AFF, S_aff=S_AFF,
        lbw=LBW, ubw=UBW, data_path=data_path
    )


if __name__=='__main__':
    main()
