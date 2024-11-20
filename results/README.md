# Getting Results

## Before running anything
Before running any scripts, set the proper absolute paths to `data` and `figures` in `scripts/consts/paths.py`.


## Accessing the dataset
Unpack the contents of [this zip](https://ucirvine-my.sharepoint.com/:u:/r/personal/dcopp_ad_uci_edu/Documents/research/2025_IEEE_Sustech_data/Efficient_%20Estimation_of_Relaxed_Model%20Parameters_Data.zip?csf=1&web=1&e=KE2jPM) (`crazyflie`, `fusion_one`) to `data`.


## Setting simulation trial parameters
To set the number of desired simulation trials, edit `NUM_TRIALS` in `scripts/consts/trials.py`. Other simulation trial, controller, and estimator parameters can be set in the other modules in `scripts/consts`.


## Running simulation trials
Run either `scripts/run_crazyflie_trials.py` or `scripts/run_fusion_one_trials.py`.


## Getting plots and results
Run `scripts/get_plots.py`.
