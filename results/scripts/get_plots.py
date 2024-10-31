import pickle
from os import listdir
from os.path import isfile, join
from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "Times New Roman"
})


from par.experiments.data import *

from consts.trials import SOLVERS
from consts.plots import *


def get_datasets(
    data_paths: List[str]
) -> List[List[TrialData]]:
    datasets_per_solver = []

    for data_path in data_paths:
        datasets_per_traj = []
        file_paths = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]

        for file_path in file_paths:
            with open(file_path, 'rb') as file:
                # Load list of TrialData objects
                datasets_per_traj += [pickle.load(file)]

        datasets_per_solver += [datasets_per_traj]
        print('Finished loading data from', data_path)

    return datasets_per_solver


def get_solve_time_plot(
    model_label: str,
    plot_axis: plt.axes,
    datasets_per_solver: List[List[List[TrialData]]]
) -> None:
    solve_times_per_solver = []
    for datasets_per_traj in datasets_per_solver:
        datasets_per_traj_combined = [
            data
            for data_per_timestep in datasets_per_traj
            for data in data_per_timestep
        ]
        solve_times_per_solver += [
            get_mhpe_solve_times(datasets_per_traj_combined)
        ]

    vp = plot_axis.violinplot(solve_times_per_solver, showmedians=True, showmeans=False)

    for part in ('cmedians', 'cmins', 'cmaxes'):
        vp[part].set_color(COLORS)
    for color_index, body in enumerate(vp['bodies']):
        body.set_edgecolor(COLORS[color_index])
        body.set_facecolor('light' + COLORS[color_index])

    plot_axis.set_xticks(range(1, 1 + len(X_TICK_LABELS)), labels=X_TICK_LABELS)
    plot_axis.set_xlabel(model_label)
    plot_axis.set_ylabel('Solve Time (s)')
    plot_axis.set_yscale('log')


def get_trajectory_cost_plot(
    model_label: str,
    plot_axis: plt.axes,
    datasets_per_solver: List[List[List[TrialData]]]
) -> None:
    costs_per_solver = [
        np.array([
            get_cost(data_per_timestep) for data_per_timestep in datasets_per_traj])
        for datasets_per_traj in datasets_per_solver
    ]

    vp = plot_axis.violinplot(costs_per_solver, showmedians=True, showmeans=False)

    for part in ('cmedians', 'cmins', 'cmaxes'):
        vp[part].set_color(COLORS)
    for color_index, body in enumerate(vp['bodies']):
        body.set_edgecolor(COLORS[color_index])
        body.set_facecolor('light' + COLORS[color_index])

    plot_axis.set_xticks(range(1, 1 + len(X_TICK_LABELS)), labels=X_TICK_LABELS)
    plot_axis.set_xlabel(model_label)
    plot_axis.set_ylabel('Trajectory Cost')
    plot_axis.set_yscale('log')


def get_plots_per_model(
    model_label: str,
    solve_time_axis: plt.axes,
    traj_cost_axis: plt.axes,
    data_path: str,
) -> None:
    data_paths = [data_path + plugin + '/' for plugin in SOLVERS.keys()]
    datasets_per_solver = get_datasets(data_paths)

    get_solve_time_plot(model_label, solve_time_axis, datasets_per_solver)
    get_trajectory_cost_plot(model_label, traj_cost_axis, datasets_per_solver)
