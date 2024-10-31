from par.dynamics.vectors import *
from par.dynamics.models import NonlinearQuadrotorModel
from par.optimization import NMPC, MHPE
from par.experiments.trials import run_parallel_trials
from par.experiments.random import *

from consts.trials import *


def run_trials(
    nominal_model: NonlinearQuadrotorModel,
    M: int,
    P: np.ndarray,
    S: np.ndarray,
    P_aff: np.ndarray,
    S_aff: np.ndarray,
    lbw: np.ndarray,
    ubw: np.ndarray,
    data_path: str,
) -> None:
    true_models = []
    random_states = []
    process_noises = []

    # Generate random sim conditions for all solvers
    for i in range(NUM_TRIALS):
        true_models += [
            get_random_model(nominal_model, LB_THETA_FACTOR, UB_THETA_FACTOR)]
        random_states += [
            get_random_state(LB_POS, UB_POS, LB_VEL, UB_VEL)]
        process_noises += [
            get_process_noise_seed(lbw, ubw, SIM_LEN)]

    lb_theta, ub_theta = get_parameter_bounds(
        nominal_model.parameters, LB_THETA_FACTOR, UB_THETA_FACTOR)

    for plugin, is_qp in SOLVERS.items():
            print('Starting trials for', plugin, '...')

            full_path = data_path + plugin + '/'
            if is_qp['is_qp']:
                nmpc = NMPC(DT, N, Q, R, QF, nominal_model.as_affine())
                mhpe = MHPE(DT, M, P_aff, S_aff, nominal_model.as_affine(), plugin=plugin)
            else:
                nmpc = NMPC(DT, N, Q, R, QF, nominal_model)
                mhpe = MHPE(DT, M, P, S, nominal_model, plugin=plugin)

            run_parallel_trials(
                is_affine=is_qp['is_qp'], data_path=full_path,
                dt=DT, nmpc=nmpc, mhpe=mhpe,
                lb_theta=lb_theta, ub_theta=ub_theta,
                nominal_model=nominal_model, true_models=true_models,
                random_states=random_states, process_noises=process_noises
            )

            print('Trials for', plugin, 'completed!')
