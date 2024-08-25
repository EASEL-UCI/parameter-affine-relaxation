#!/usr/bin/python3

import casadi as cs

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

def symbol(config: dict, id: str) -> cs.SX:
    return cs.SX.sym(id, config[id]["dimensions"])
