import numpy as np


dt = 0.05
SIM_LEN = 200

MPC_N = 20
MHE_M = 10

PARAM_PERTURB_MIN = 0.5
PARAM_PERTURB_MAX = 1.5

NOISE_MIN = np.hstack((-0.5 * np.ones(7), -5.0 * np.ones(6)))
NOISE_MAX = np.hstack((0.5 * np.ones(7), 5.0 * np.ones(6)))

POS_MIN = -10 * np.ones(3)
POS_MAX = 10 * np.ones(3)

VEL_MIN = -10 * np.ones(3)
VEL_MAX = 10 * np.ones(3)

