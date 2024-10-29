from typing import Callable, List
import multiprocessing as mp
import datetime
import pickle

import numpy as np

from par.dynamics.vectors import State, Input, ProcessNoise, VectorList, \
                                    ModelParameters
from par.dynamics.models import NonlinearQuadrotorModel
from par.dynamics.vectors import get_affine_parameter_bounds
from par.optimization import NMPC, MHPE
from par.utils.experiments.data import SimData


def run_parallel_trials(
    trial_func: Callable,
    nmpc: NMPC,
    mhpe: MHPE,
    nominal_model: NonlinearQuadrotorModel,
    true_model: NonlinearQuadrotorModel,
    lb_theta: ModelParameters,
    ub_theta: ModelParameters,
    random_states: List[State],
    process_noises: List[VectorList],
    data_path: str,
) -> None:
    assert len(random_states) == len(process_noises)
    trial_args = [
        [
            nmpc, mhpe, nominal_model, true_model, lb_theta, ub_theta,
            random_states[i], process_noises[i], data_path
        ]
        for i in range(len(random_states))
    ]

    p = mp.Pool()
    p.map(trial_func, trial_args)


def affine_adaptive_mpc_trial(
    nmpc: NMPC,
    mhpe: MHPE,
    nominal_model: NonlinearQuadrotorModel,
    true_model: NonlinearQuadrotorModel,
    lb_theta: ModelParameters,
    ub_theta: ModelParameters,
    random_state: State,
    process_noises: VectorList,
    dt: float,
    data_path: str,
) -> None:
    # Get affine-in-parameter models
    nominal_model_aff = nominal_model.as_affine()

    # Init state
    x = random_state

    # MHE stuff
    lb_theta_aff, ub_theta_aff = get_affine_parameter_bounds(lb_theta, ub_theta)

    # MPC args
    theta = nominal_model.parameters.as_affine()
    xref = VectorList( nmpc.N * [State()] )
    uref = VectorList( nmpc.n * [Input()] )
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
            lbu=nominal_model_aff.lbu, ubu=nominal_model_aff.ubu,
            xs_guess=xs_guess, us_guess=us_guess
        )
        xs_guess = nmpc.get_predicted_states()
        us_guess = nmpc.get_predicted_inputs()
        u = us_guess.get(0)

        # Generate uniform noise on the acceleration
        w = process_noises.get(k).as_array()

        # Update current state and trajectory history
        x = true_model.step_sim(dt=dt, x=x, u=u, w=w)
        xs.append(x)
        us.append(u)

        # Get parameter estimate
        mhpe.solve(x, u, w, lb_theta=lb_theta_aff, ub_theta=ub_theta_aff)
        theta = mhpe.get_parameter_estimate()

        # log sim data
        data += [SimData(
            x, u, w, theta, xref, uref, true_model.parameters.as_affine(),
            nmpc.Q, nmpc.R, mhpe.get_full_solution(), mhpe.get_solver_stats(),
        )]

        print(f'\ninput {k}: \n{u.as_array()}')
        print(f'\n\n\nstate {k+1}: \n{x.as_array()}')
        print(f'\naffine parameter estimate {k+1}: \n{theta.as_array()}')
    print(f'\ntrue affine parameter: \n{true_model.parameters.as_affine().as_array()}')
    print(f'\nnominal affine parameter: \n{nominal_model.parameters.as_affine().as_array()}')

    file_path = data_path + '{date:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

    theta_acc = true_model.parameters.as_affine().as_array()
    normalized_errors = np.zeros(len(theta_acc))
    for i in range(true_model.as_affine().ntheta):
        normalized_errors[i] = ( theta.as_array()[i] - theta_acc[i] ) / np.abs(theta_acc[i])

    print(f'\nNormalized parameter estimate errors: \n{normalized_errors}')
    print(f'\nNorm of parameter estimate error: {np.linalg.norm(normalized_errors)}')
