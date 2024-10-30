from typing import Callable, List
import multiprocessing as mp
import datetime
import pickle

import numpy as np

from par.dynamics.vectors import *
from par.dynamics.models import NonlinearQuadrotorModel
from par.optimization import NMPC, MHPE
from par.utils.experiments.data import SimData


def run_parallel_trials(
    is_affine: bool,
    data_path: str,
    dt: int,
    nmpc: NMPC,
    mhpe: MHPE,
    lb_theta: ModelParameters,
    ub_theta: ModelParameters,
    nominal_model: NonlinearQuadrotorModel,
    true_models: List[NonlinearQuadrotorModel],
    random_states: List[State],
    process_noises: List[VectorList],
) -> None:
    assert len(random_states) == len(process_noises)
    trial_args = [
        {
            'is_affine': is_affine, 'data_path': data_path, 'dt': dt,
            'nmpc': nmpc, 'mhpe': mhpe, 'lb_theta': lb_theta, 'ub_theta': ub_theta,
            'nominal_model': nominal_model, 'true_model': true_models[i],
            'random_state': random_states[i], 'process_noises': process_noises[i],
        }
        for i in range(len(random_states))
    ]
    p = mp.Pool()
    p.map(adaptive_mpc_trial, trial_args)


def adaptive_mpc_trial(
    trial_args: dict
) -> None:
    # Unpack args
    is_affine = trial_args['is_affine']
    data_path = trial_args['data_path']
    dt = trial_args['dt']
    nmpc = trial_args['nmpc']
    mhpe = trial_args['mhpe']
    lb_theta = trial_args['lb_theta']
    ub_theta = trial_args['ub_theta']
    nominal_model = trial_args['nominal_model']
    true_model = trial_args['true_model']
    random_state = trial_args['random_state']
    process_noises = trial_args['process_noises']

    # Get affine-in-parameter models
    if is_affine:
        nominal_model = nominal_model.as_affine()
        lb_theta, ub_theta = get_affine_parameter_bounds(lb_theta, ub_theta)

    # Init state
    x = random_state
    mhpe.reset_measurements(x)

    # MPC args
    theta = nominal_model.parameters
    xref = VectorList( nmpc.N * [State()] )
    uref = VectorList( nmpc.N * [Input()] )
    xs_guess = None
    us_guess = None

    # Sim stuff
    w = ProcessNoise()
    xs = VectorList()
    us = VectorList()
    data = []

    # Iterate sim
    for k in range(len(process_noises.get())):
        # Solve, update warmstarts, and get the control input
        nmpc.solve(
            x, xref, uref, theta,
            lbu=nominal_model.lbu, ubu=nominal_model.ubu,
            xs_guess=xs_guess, us_guess=us_guess
        )
        xs_guess = nmpc.get_predicted_states()
        us_guess = nmpc.get_predicted_inputs()
        u = us_guess.get(0)

        # Generate uniform noise on the acceleration
        w = process_noises.get(k)

        # Update current state and trajectory history
        x = true_model.step_sim(dt=dt, x=x, u=u, w=w)
        xs.append(x)
        us.append(u)

        # Get parameter estimate
        mhpe.solve(x, u, w, lb_theta=lb_theta, ub_theta=ub_theta)
        theta = mhpe.get_parameter_estimate()

        # log sim data
        if k >= mhpe.M:
            data += [SimData(
                x, u, w, theta, xref, uref, true_model.parameters.as_affine(),
                nmpc.Q, nmpc.R, mhpe.get_full_solution(), mhpe.get_solver_stats(),
            )]

    file_path = data_path + str(datetime.datetime.now()) + '.pkl'
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

    print('Trial completed!')
