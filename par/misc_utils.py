from typing import Any


def is_none(a: Any) -> bool:
    if type(a) == type(None):
        return True
    else:
        return False
