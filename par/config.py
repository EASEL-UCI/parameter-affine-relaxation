import numpy as np

from par.constants import BIG_NEGATIVE, BIG_POSITIVE, GRAVITY


PARAMETER_CONFIG = {
    "m": {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "a": {
        "dimensions": 3,
        "lower_bound": np.zeros(3),
        "upper_bound": BIG_POSITIVE * np.zeros(3),
        "default_value": np.zeros(3),
    },
    "Ixx": {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "Iyy": {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "Izz": {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "b": {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
}


RELAXED_PARAMETER_CONFIG = {
    "M": {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "A": {
        "dimensions": 3,
        "lower_bound": np.zeros(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "IXX": {
        "dimensions": 1,
        "lower_bound": BIG_NEGATIVE,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "IYY": {
        "dimensions": 1,
        "lower_bound": BIG_NEGATIVE,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "IZZ": {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "IXX_rb": {
        "dimensions": 1,
        "lower_bound": BIG_NEGATIVE,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "IYY_rb": {
        "dimensions": 1,
        "lower_bound": BIG_NEGATIVE,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "IZZ_rb": {
        "dimensions": 1,
        "lower_bound": BIG_NEGATIVE,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
}


STATE_CONFIG = {
    "POSITION": {
        "dimensions": 3,
        "lower_bound": np.hstack((BIG_NEGATIVE * np.ones(3))),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "ATTITUDE": {
        "dimensions": 4,
        "lower_bound": -1.0 * np.ones(4),
        "upper_bound": 1.0 * np.ones(4),
        "default_value": np.hstack((1.0, np.zeros(3))),
    },
    "BODY_FRAME_LINEAR_VELOCITY": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "BODY_FRAME_ANGULAR_VELOCITY": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
}


KOOPMAN_STATE_CONFIG = {
    "BODY_FRAME_POSITION": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "BODY_FRAME_LINEAR_VELOCITY": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "BODY_FRAME_GRAVITY": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.array([0.0, 0.0, -GRAVITY]),
    },
    "BODY_FRAME_ANGULAR_VELOCITY": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
}


INPUT_CONFIG = {
    "THRUSTS": {
        "dimensions": 4,
        "lower_bound": np.zeros(4),
        "upper_bound": BIG_POSITIVE * np.ones(4),
        "default_value": np.zeros(4),
    },
}


PROCESS_NOISE_CONFIG = {
    "BODY_FRAME_LINEAR_VELOCITY": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "ATTITUDE_RATE": {
        "dimensions": 4,
        "lower_bound": BIG_NEGATIVE * np.ones(4),
        "upper_bound": BIG_POSITIVE * np.ones(4),
        "default_value": np.zeros(4),
    },
    "BODY_FRAME_LINEAR_ACCELERATION": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "BODY_FRAME_ANGULAR_ACCELERATION": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
}


KOOPMAN_PROCESS_NOISE_CONFIG = {
    "PROCESS_NOISE": {
        "dimensions": 12,
        "lower_bound": BIG_NEGATIVE * np.ones(13),
        "upper_bound": BIG_POSITIVE * np.ones(13),
    },
}


QP_SOLVER_CONFIG = {
    "qpoases": "qpoases",
    "osqp": "osqp",
    "proxqp": "proxqp",
}
