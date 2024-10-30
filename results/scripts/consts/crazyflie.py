import numpy as np


# Nonlinear MHE weights
P = np.diag(np.hstack((
    1.0e2,
    1.0e1 * np.ones(3),
    1.0e5 * np.ones(3),
    1.0e0 * np.ones(4),
    1.0e2 * np.ones(4),
    1.0e2 * np.ones(4),
    1.0e2 * np.ones(4)
)))
S = 1.0e3 * np.eye(13)

# Linear-quadratic MHE weights
P_AFF = np.diag(np.hstack((
    1.0e-1,
    np.ones(3),
    1.0e-3 * np.ones(4),
    1.0e-3 * np.ones(4),
    1.0e-3 * np.ones(4),
    1.0e1 * np.ones(3),
)))
S_AFF = 1.0e3 * np.eye(13)

# process noise constants
LBW = np.hstack((np.zeros(7), -5.0 * np.ones(6)))
UBW = np.hstack((np.zeros(7), 5.0 * np.ones(6)))
