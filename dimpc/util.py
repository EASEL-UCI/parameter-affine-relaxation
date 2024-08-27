#!/usr/bin/python3

from typing import List

import casadi as cs
import numpy as np

BIG_POSITIVE_NUMBER = 10**9
BIG_NEGATIVE_NUMBER = -10**9

def get_sub_config(config: dict, id: str) -> dict:
    sub_config = {}
    for config_id, config_member in config.items():
        if config_id.find(id) == -1:
            continue
        else:
            sub_config[config_id] = config_member
    return sub_config

def get_dimensions(config: dict, id: str) -> int:
    sub_config = get_sub_config(config, id)
    dims = 0
    for config_member in sub_config.values():
        dims += config_member["dimensions"]
    return dims

def get_all_dimensions(config: dict) -> int:
    dims = 0
    for config_member in config.values():
        dims += config_member["dimensions"]
    return dims

def get_start_or_end_index(config: dict, id: str, is_start: bool) -> int:
    i = 0
    for config_id in config.keys():
        if not is_start:
            i += config[config_id]["dimensions"]
        if config_id == id:
            return i
        elif config_id == list(config.keys())[-1]:
            raise IndexError
        elif is_start:
            i += config[config_id]["dimensions"]

def get_start_index(config: dict, id: str) -> int:
    return get_start_or_end_index(config=config, id=id, is_start=True)

def get_stop_index(config: dict, id: str) -> int:
    return get_start_or_end_index(config=config, id=id, is_start=False)

def symbolic(config: dict, id: str) -> cs.SX:
    return cs.SX.sym(id, config[id]["dimensions"])

def get_default_vector(config: dict, dimensions: int) -> List:
    i = 0
    vector = []
    for config_id in config.keys():
        vector += config[config_id]["default_value"]
        i += config[config_id]["dimensions"]
        if i > dimensions:
            raise IndexError
        elif i == dimensions:
            return vector
    raise IndexError

def is_integer(config: dict, dimensions: int) -> List:
    i = 0
    is_integer = []
    for config_id in config.keys():
        if config[config_id]["member_type"] != int:
            is_integer += [False] * config[config_id]["dimensions"]
        else:
            is_integer += [True] * config[config_id]["dimensions"]
        i += config[config_id]["dimensions"]
        if i > dimensions:
            raise IndexError
        elif i == dimensions:
            return is_integer
    raise IndexError

def subtract_lists(x: List[float], y: List[float]) -> List[float]:
    return (np.array(x) - np.array(y)).tolist()
