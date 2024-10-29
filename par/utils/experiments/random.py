import numpy as np

from par.dynamics.vectors import ProcessNoise, VectorList, State
from par.utils.math import random_unit_quaternion


def get_process_noise_seed(
    lbw: np.ndarray,
    ubw: np.ndarray,
    sim_len: int,
) -> VectorList:
    ws = VectorList()
    for i in range(sim_len):
        w_arr = np.random.uniform(lbw, ubw)
        ws.append(ProcessNoise(w_arr))
    return ws


def get_random_state(
    lb_pos: np.ndarray,
    ub_pos: np.ndarray,
    lb_vel: np.ndarray,
    ub_vel: np.ndarray,
) -> State:
    x = State()
    x['position_wf'] = np.random.uniform(lb_pos, ub_pos)
    x['attitude'] = random_unit_quaternion()
    x['linear_velocity_bf'] = np.random.uniform(lb_vel, ub_vel)
    x['angular_velocity_bf'] = np.random.uniform(lb_vel, ub_vel)
    return x
