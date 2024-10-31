import matplotlib.pyplot as plt

from get_plots import get_plots_per_model
from consts.trials import SOLVERS
from consts.paths import DATA_PATH, CRAZYFLIE_PATH


def main():
    fig, axs = plt.subplots(2, 3)

    # Add Crazyflie plots
    data_path = DATA_PATH + CRAZYFLIE_PATH
    get_plots_per_model('Crazyflie', axs[0,0], axs[1,0], data_path)

    # Add Fusion One plots

    # Add Asymmetric Qadrotor plots

    for ax in axs.flat:
        ax.label_outer()
    plt.show()

if __name__=='__main__':
    main()
