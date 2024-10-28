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
        'upper_bound': BIG_POSITIVE * np.ones(3),
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
    'c': {
        'dimensions': 4,
        'lower_bound': np.zeros(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    'r': {
        'dimensions': 4,
        'lower_bound': BIG_NEGATIVE * np.ones(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    's': {
        'dimensions': 4,
        'lower_bound': BIG_NEGATIVE * np.ones(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
}


RELAXED_PARAMETER_CONFIG = {
    '1/m': {
        'dimensions': 1,
        'lower_bound': 0.0,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'a/m': {
        'dimensions': 3,
        'lower_bound': np.zeros(3),
        'upper_bound': BIG_POSITIVE * np.ones(3),
        'default_value': np.zeros(3),
    },
    's/Ixx': {
        'dimensions': 4,
        'lower_bound': BIG_NEGATIVE * np.ones(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    'r/Iyy': {
        'dimensions': 4,
        'lower_bound': BIG_NEGATIVE * np.ones(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    'c/Izz': {
        'dimensions': 4,
        'lower_bound': np.zeros(4),
        'upper_bound': BIG_POSITIVE * np.ones(4),
        'default_value': np.zeros(4),
    },
    'Ixx_cross': {
        'dimensions': 1,
        'lower_bound': BIG_NEGATIVE,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'Iyy_cross': {
        'dimensions': 1,
        'lower_bound': BIG_NEGATIVE,
        'upper_bound': BIG_POSITIVE,
        'default_value': 0.0,
    },
    'Izz_cross': {
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
    'normalized_squared_motor_speed': {
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
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.sb': 'yes',
    },
}


QP_SOLVER_CONFIG = {
    'qpoases': {},
    'osqp': {
        'osqp.max_iter': 4000,
    },
    'proxqp': {
        'proxqp.max_iter': 1000,
    },
}
