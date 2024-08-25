#!/usr/bin/python3


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
