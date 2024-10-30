import numpy as np


NUM_TRIALS = 100

DT = 0.05
SIM_LEN = 200

LB_THETA_FACTOR = 0.6
UB_THETA_FACTOR = 1.4

LB_POS = -10 * np.ones(3)
UB_POS = 10 * np.ones(3)

LB_VEL = -5 * np.ones(3)
UB_VEL = 5 * np.ones(3)

SOLVERS = {
    'proxqp': {'is_qp': True},
    'osqp': {'is_qp': True},
    'ipopt': {'is_qp': False},
}

# MHPE constants
M = 10

# NMPC constants
N = 10
Q = np.diag(np.hstack((np.ones(10), 1.0e-2 * np.ones(3))))
R = np.eye(4)
QF = 2.0 * Q
