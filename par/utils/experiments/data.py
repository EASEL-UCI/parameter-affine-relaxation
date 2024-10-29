from typing import Union, List, Tuple

import numpy as np

from par.dynamics.vectors import State, Input, ProcessNoise, ModelParameters, \
                                    AffineModelParameters


class SimData():
    def __init__(
        self,
        x: State,
        u: Input,
        w: ProcessNoise,
        theta: Union[ModelParameters, AffineModelParameters],
        xref: State,
        uref: Input,
        theta_true: Union[ModelParameters, AffineModelParameters],
        Q: np.ndarray,
        R: np.ndarray,
        mhpe_solution: dict,
        mhpe_solver_stats: dict,
    ) -> None:
        self.x = x
        self.u = u
        self.w = w
        self.theta = theta
        self.xref = xref
        self.uref = uref
        self.theta_true = theta_true
        self.Q = Q
        self.R = R
        self.mhpe_solution = mhpe_solution
        self.mhpe_solver_stats = mhpe_solver_stats


def get_mhpe_solve_times(data: List[SimData]) -> np.ndarray:
    try:
        return np.array([
            data_k.mhpe_solver_stats['t_wall_total'] for data_k in data])
    except KeyError:
        return np.array([
            data_k.mhpe_solver_stats['t_wall_solver'] for data_k in data])


def get_average_mhpe_solve_time(data: List[SimData]) -> float:
    return np.average(get_mhpe_solve_times(data))


def get_mhpe_solve_time_quartiles(data: List[SimData]) -> Tuple[float]:
    solve_times = get_mhpe_solve_times(data)
    Q1 = np.quantile(solve_times, 0.25)
    Q2 = np.quantile(solve_times, 0.50)
    Q3 = np.quantile(solve_times, 0.75)
    return Q1, Q2, Q3


def get_mhpe_non_convergences(data: List[SimData]) -> int:
    non_convergences = 0
    for data_k in data:
        if not data_k.mhpe_solver_stats['success']:
            non_convergences += 1
    return non_convergences


def get_states(data: List[SimData]) -> np.ndarray:
    return np.array([data_k.x.as_array() for data_k in data])


def get_inputs(data: List[SimData]) -> np.ndarray:
    return np.array([data_k.u.as_array() for data_k in data])


def get_cost(data: List[SimData]) -> float:
    xs = get_states(data)
    us = get_inputs(data)
    xrefs = [data_k.xref.as_array() for data_k in data]
    urefs = [data_k.uref.as_array() for data_k in data]
    Qs = [data_k.Q for data_k in data]
    Rs = [data_k.R for data_k in data]
    cost = 0.0
    for i in range(len(data)):
        x_err = xs[i] - xrefs[i]
        u_err = us[i] - urefs[i]
        cost += x_err @ Qs[i] @ x_err + u_err @ Rs[i] @ u_err
    return cost
