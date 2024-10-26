import numpy as np

from par.constants import BIG_NEGATIVE, BIG_POSITIVE, GRAVITY


PARAMETER_CONFIG = {
    'm': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'a': {
        'dimensions': 3,
        'lower_bound': np.zeros(3),
        'upper_bound': BIG_POSITIVE * np.zeros(3),
        'default_value': np.zeros(3),
    },
    'Ixx': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'Iyy': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'Izz': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'b': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
}


RELAXED_PARAMETER_CONFIG = {
    'M': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'A': {
        'dimensions': 3,
        'lower_bound': np.zeros(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'IXX': {
        'dimensions': 1,
        'lower_bound': BIG_NEGATIVE,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'IYY': {
        'dimensions': 1,
        'lower_bound': BIG_NEGATIVE,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'IZZ': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'IXX_rb': {
        'dimensions': 1,
        'lower_bound': BIG_NEGATIVE,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'IYY_rb': {
        'dimensions': 1,
        'lower_bound': BIG_NEGATIVE,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'IZZ_rb': {
        'dimensions': 1,
        'lower_bound': BIG_NEGATIVE,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
}


STATE_CONFIG = {
    'position_wf': {
        'dimensions': 3,
        'lower_bound': np.hstack((BIG_NEGATIVE * np.ones(3))),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'attitude': {
        'dimensions': 4,
        'lower_bound': -1.0 * np.ones(4),
        'upper_bound': 1.0 * np.ones(4),
        'default_value': np.hstack((1.0, np.zeros(3))),
    },
    'linear_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'angular_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
}


KOOPMAN_STATE_CONFIG = {
    'position_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'linear_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'gravity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.array([0.0, 0.0, -GRAVITY]),
    },
    'angular_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
}


INPUT_CONFIG = {
    'thrusts': {
        'dimensions': 4,
        'lower_bound': np.zeros(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
}


PROCESS_NOISE_CONFIG = {
    'linear_velocity_wf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'attitude_rate': {
        'dimensions': 4,
        'lower_bound': BIG_NEGATIVE * np.ones(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    'linear_acceleration_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'angular_acceleration_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
}


KOOPMAN_PROCESS_NOISE_CONFIG = {
    'linear_velocity_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'linear_acceleration_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    'gravity_rate_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.array([0.0, 0.0, -GRAVITY]),
    },
    'angular_acceleration_bf': {
        'dimensions': 3,
        'lower_bound': BIG_NEGATIVE * np.ones(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
}


NLP_SOLVER_CONFIG = {
    'ipopt': {
        'ipopt.max_iter': 1000,
    }
}


QP_SOLVER_CONFIG = {
    'qpoases': {
        'qpoases.max_iter': 1000,
    },
    'osqp': {
        'osqp.check_termination': 1000,
        'osqp.max_iter': 1000,
    },
    'proxqp': {
        'proxqp.max_iter': 1000
    },
}
