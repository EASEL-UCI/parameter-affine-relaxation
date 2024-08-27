#!/usr/bin/python3


from dimpc.util import get_dimensions, get_all_dimensions

CONTACT_ACTIVATION_FACTOR = 1000.0
CONTACT_SPRING_FACTOR = 1000.0
CONTACT_DAMPER_FACTOR = 1000.0
GRAVITY = 9.81

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
    "xB":   {"type": list, "member_type": float, "dimensions": 4},
    "yB":   {"type": list, "member_type": float, "dimensions": 4},
}

STATE_CONFIG = {
    "DRONE_POSITION": {
        "type": list, "member_type": float, "dimensions": 3,
        "lower_bound": [None, None, 0.0], "upper_bound": [None, None, None],
        "default_value": [0.0, 0.0, 0.0,],
    },
    "DRONE_ORIENTATION": {
        "type": list, "member_type": float, "dimensions": 4,
        "lower_bound": [0.0, 0.0, 0.0, 0.0], "upper_bound": [1.0, 1.0, 1.0, 1.0],
        "default_value": [1.0, 0.0, 0.0, 0.0,],
    },
    "DRONE_LINEAR_VELOCITY": {
        "type": list, "member_type": float, "dimensions": 3,
        "lower_bound": [None, None, None], "upper_bound": [None, None, None],
        "default_value": [0.0, 0.0, 0.0,],
    },
    "DRONE_ANGULAR_VELOCITY": {
        "type": list, "member_type": float, "dimensions": 3,
        "lower_bound": [None, None, None], "upper_bound": [None, None, None],
        "default_value": [0.0, 0.0, 0.0,],
    },
    "PAYLOAD_POSITION_0": {
        "type": list, "member_type": float, "dimensions": 3,
        "lower_bound": [None, None, None], "upper_bound": [None, None, None],
        "default_value": [0.0, 0.0, 0.0,],
    },
    "PAYLOAD_VELOCITY_0": {
        "type": list, "member_type": float, "dimensions": 3,
        "lower_bound": [None, None, None], "upper_bound": [None, None, None],
        "default_value": [0.0, 0.0, 0.0,],
    },
}

INPUT_CONFIG = {
    "DRONE_THRUSTS": {
        "type": list, "member_type": float, "dimensions": 4,
        "lower_bound": [0.0, 0.0, 0.0, 0.0], "upper_bound": [None, None, None],
        "default_value": [0.0, 0.0, 0.0, 0.0,],
    },
    "PAYLOAD_RELEASE": {
        "type": list, "member_type": int, "dimensions": 1,
        "lower_bound": [0], "upper_bound": [1],
        "default_value": [0],
    },
}

MODEL_TYPE_CONFIG = {
    "DRONE": {
        "state": {"dimensions": get_dimensions(config=STATE_CONFIG, id="DRONE")},
        "input": {"dimensions": get_dimensions(config=INPUT_CONFIG, id="DRONE")},
    },
    "PAYLOAD": {
        "state": {"dimensions": get_all_dimensions(config=STATE_CONFIG)},
        "input": {"dimensions": get_all_dimensions(config=INPUT_CONFIG)},
    },
}
