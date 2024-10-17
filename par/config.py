import numpy as np

from par.constants import BIG_NEGATIVE, BIG_POSITIVE, GRAVITY


PARAMETER_CONFIG = {
    "m":     {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "a":     {
        "dimensions": 3,
        "lower_bound": np.zeros(3),
        "upper_bound": BIG_POSITIVE * np.zeros(3),
        "default_value": np.zeros(3),
    },
    "Ixx":   {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "Iyy":   {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "Izz":   {
        "dimensions": 1,
        "lower_bound": 0.0,
        "upper_bound": BIG_POSITIVE,
        "default_value": 0.0,
    },
    "k":     {
        "dimensions": 4,
        "lower_bound": np.zeros(4),
        "upper_bound": BIG_POSITIVE * np.zeros(4),
        "default_value": np.zeros(4),
    },
    "c":     {
        "dimensions": 4,
        "lower_bound": np.zeros(4),
        "upper_bound": BIG_POSITIVE * np.zeros(4),
        "default_value": np.zeros(4),
    },
    "r":     {
        "dimensions": 4,
        "lower_bound": BIG_NEGATIVE * np.ones(4),
        "upper_bound": BIG_POSITIVE * np.ones(4),
        "default_value": np.zeros(4),
    },
    "s":     {
        "dimensions": 4,
        "lower_bound": BIG_NEGATIVE * np.ones(4),
        "upper_bound": BIG_POSITIVE * np.ones(4),
        "default_value": np.zeros(4),
    },
}


RELAXED_PARAMETER_CONFIG = {
    "A":     {
        "dimensions": 3,
        "lower_bound": np.zeros(3),
        "upper_bound": BIG_POSITIVE * np.zeros(3),
    },
    "K":     {
        "dimensions": 4,
        "lower_bound": np.zeros(4),
        "upper_bound": BIG_POSITIVE * np.zeros(4),
    },
    "S":   {
        "dimensions": 4,
        "lower_bound": BIG_NEGATIVE * np.ones(4),
        "upper_bound": BIG_POSITIVE * np.ones(4),
    },
    "R":   {
        "dimensions": 4,
        "lower_bound": BIG_NEGATIVE * np.ones(4),
        "upper_bound": BIG_POSITIVE * np.ones(4),
    },
    "C":   {
        "dimensions": 4,
        "lower_bound": np.zeros(4),
        "upper_bound": BIG_POSITIVE * np.zeros(4)
    },
    "IXX":     {
        "dimensions": 1,
        "lower_bound": BIG_NEGATIVE,
        "upper_bound": BIG_POSITIVE,
    },
    "IYY":     {
        "dimensions": 1,
        "lower_bound": BIG_NEGATIVE,
        "upper_bound": BIG_POSITIVE,
    },
    "IZZ":     {
        "dimensions": 1,
        "lower_bound": BIG_NEGATIVE,
        "upper_bound": BIG_POSITIVE,
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


KOOPMAN_CONFIG = {
    "BODY_FRAME_POSITION": {
        "dimensions": 3,
        "lower_bound": np.hstack((BIG_NEGATIVE * np.ones(3))),
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
    "SQUARED_MOTOR_ANGULAR_VELOCITY": {
        "dimensions": 4,
        "lower_bound": np.zeros(4),
        "upper_bound": BIG_POSITIVE * np.ones(4),
        "default_value": np.zeros(4),
    },
}


NOISE_CONFIG = {
    "PROCESS_NOISE": {
        "dimensions": 13,
        "lower_bound": BIG_NEGATIVE * np.ones(13),
        "upper_bound": BIG_POSITIVE * np.ones(13),
    },
}
