#!/usr/bin/python3

import numpy as np

from mitodd.helpers import get_dimensions, get_all_dimensions


PARAMETER_CONFIG = {
    "m":    {"type": float},
    "Ixx":  {"type": float},
    "Iyy":  {"type": float},
    "Izz":  {"type": float},
    "Ax":   {"type": float},
    "Ay":   {"type": float},
    "Az":   {"type": float},
    "kf":   {"type": float},
    "km":   {"type": float},
    "umin": {"type": float},
    "umax": {"type": float},
    "xB":   {"type": np.ndarray, "member_type": np.float64, "dimensions": 4},
    "yB":   {"type": np.ndarray, "member_type": np.float64, "dimensions": 4},
}


STATE_VECTOR_CONFIG = {
    "DRONE_POSITION":         {"type": np.ndarray, "member_type": np.float64, "dimensions": 3},
    "DRONE_ORIENTATION":      {"type": np.ndarray, "member_type": np.float64, "dimensions": 4},
    "DRONE_LINEAR_VELOCITY":  {"type": np.ndarray, "member_type": np.float64, "dimensions": 3},
    "DRONE_ANGULAR_VELOCITY": {"type": np.ndarray, "member_type": np.float64, "dimensions": 3},
    "PAYLOAD_POSITION_0":     {"type": np.ndarray, "member_type": np.float64, "dimensions": 3},
    "PAYLOAD_VELOCITY_0":     {"type": np.ndarray, "member_type": np.float64, "dimensions": 3},
}


INPUT_VECTOR_CONFIG = {
    "DRONE_THRUSTS":   {"type": np.ndarray, "member_type": np.float64, "dimensions": 4},
    "PAYLOAD_RELEASE": {"type": np.ndarray, "member_type": np.int64,   "dimensions": 1},
}


MODEL_TYPES = {
    "DRONE": {
        "state": {"dimensions": get_dimensions(config=STATE_VECTOR_CONFIG, id="DRONE")},
        "input": {"dimensions": get_dimensions(config=INPUT_VECTOR_CONFIG, id="DRONE")},
    },
    "PAYLOAD": {
        "state": {"dimensions": get_all_dimensions(config=STATE_VECTOR_CONFIG)},
        "input": {"dimensions": get_all_dimensions(config=INPUT_VECTOR_CONFIG)},
    },
}
