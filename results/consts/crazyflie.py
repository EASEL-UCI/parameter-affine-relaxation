import numpy as np


# mpc constants
N = 10
Q = np.diag(np.hstack((np.ones(10), 1.0e-2 * np.ones(3))))
R = np.eye(4)
QF = 2.0 * Q

# mhe constants
M = 10
P_AFF = np.diag(np.hstack((
    1.0e-1,
    np.ones(3),
    1.0e-3 * np.ones(4),
    1.0e-3 * np.ones(4),
    1.0e-3 * np.ones(4),
    1.0e1 * np.ones(3),
)))
S = 1.0e3 * np.eye(13)

# process noise constants
LBW = np.hstack((-10.0 * np.ones(13)))
UBW = np.hstack((10.0 * np.ones(13)))
