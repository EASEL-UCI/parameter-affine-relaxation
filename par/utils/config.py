from typing import List

import casadi as cs
import numpy as np

from par.utils.misc import is_none


def get_sub_config(id: str, config: dict) -> dict:
    sub_config = {}
    for config_id, config_member in config.items():
        if config_id.find(id) == -1:
            continue
        else:
            sub_config[config_id] = config_member
    return sub_config


def get_dimensions(config: dict, copies=1) -> int:
    dims = 0
    for config_member in config.values():
        dims += config_member["dimensions"]
    return copies * dims


def get_start_or_end_index(id: str, config: dict, is_start: bool) -> int:
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


def get_start_index(id: str, config: dict) -> int:
    return get_start_or_end_index(config=config, id=id, is_start=True)


def get_stop_index(id: str, config: dict) -> int:
    return get_start_or_end_index(config=config, id=id, is_start=False)


def get_subvector(vector, id: str, config: dict) -> np.ndarray:
    return vector[get_start_index(id, config) : get_stop_index(id, config)]


def insert_subvector(vector, subvector, id: str, config: dict, copies=1) -> np.ndarray:
    start = copies * get_start_index(id, config)
    stop = copies * get_stop_index(id, config)
    vector[start : stop] = subvector
    return vector


def symbolic(id: str, config: dict, copies=1) -> cs.SX:
    return cs.SX.sym(id, copies * config[id]["dimensions"])


def get_config_values(
    id: str,
    config: dict,
    copies: int = 1
) -> np.ndarray:
    dims = get_dimensions(config)
    i = 0
    vector = []
    for config_id in config.keys():
        delta_i = config[config_id]["dimensions"]

        if delta_i == 1:
            vector += copies * [config[config_id][id]]
        elif delta_i > 1:
            vector += copies * list(config[config_id][id])

        i += delta_i
        if i > dims:
            raise IndexError
        elif i == dims:
            return np.array(vector)
    raise IndexError
