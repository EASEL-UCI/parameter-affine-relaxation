import numpy as np
from par.constants import BIG_NEGATIVE, BIG_POSITIVE


PARAMETER_CONFIG = {
    "m":     {"dimensions": 1},
    "a":     {"dimensions": 3},
    "Ixx":   {"dimensions": 1},
    "Iyy":   {"dimensions": 1},
    "Izz":   {"dimensions": 1},
    "k":     {"dimensions": 4},
    "c":     {"dimensions": 4},
    "r":     {"dimensions": 4},
    "s":     {"dimensions": 4},
}


RELAXED_PARAMETER_CONFIG = {
    "relaxed_parameters": {"dimensions": 22}
}


STATE_CONFIG = {
    "POSITION": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "ATTITUDE": {
        "dimensions": 4,
        "lower_bound": -1.0 * np.ones(4),
        "upper_bound": 1.0 * np.ones(4),
        "default_value": np.hstack((1.0, np.zeros(3))),
    },
    "BODY_LINEAR_VELOCITY": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
    "BODY_ANGULAR_VELOCITY": {
        "dimensions": 3,
        "lower_bound": BIG_NEGATIVE * np.ones(3),
        "upper_bound": BIG_POSITIVE * np.ones(3),
        "default_value": np.zeros(3),
    },
}


INPUT_CONFIG = {
    "MOTOR_SPEED_SQUARED": {
        "dimensions": 4,
        "lower_bound": np.zeros(4),
        "upper_bound": BIG_POSITIVE * np.ones(4),
        "default_value": np.zeros(4),
    },
}
