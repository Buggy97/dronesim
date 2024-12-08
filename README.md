# dronesim

## Dependencies
Used python version 3.10.12.

Install the dependencies using
`pip install -r requirements.txt`

## Description
The project has several scripts that allows the generation of drone trajectories and the related sensing and RemoteID positions datasets

### `gen_flight_plan.py`
Main script that generates all the datasets. It can be configured for a lot of parameters that affect the generation of the datasets. All of them are available are the start of the script and are self explanatory.
It generates: simulation configuration with UAV flight plans and scenario aspects including buildings and antennas, UAV trajectories, RemoteID position dataset, Sensing dataset.
The design is modular, allowing us to hardcode different aspects of the simulation such as buildings, uav flights, start times, velocities, antenna configuration, and flight plans. An explanation to do this is available in the script.
The script also has some commented examples of hardcoding that were used during experiments. They can be easily found searching for the word 'Trick'

### `association.py`
Runs the location based attack on the sensing datasets to construct a series of trajectories. This script tries to guess the trajectories based on the sensing data set. After guessing the trajectories, it tries to associated RemoteID messages to it.
The output is a file containing a series of trajectories, metrics, and statistics. For now the script outputs the following metrics along their statistics: terminal accuracy, nomalized RMSE, normalized DTW ditance, correct RemoteID associations. This script is also referred to as the location based attack.

### `cities.py`
Example of scenario that have all the hardcoded values ready to be used in the `gen_flight_plan.py` script. This file can be imported as a python module so a scenario can be referred to easily. An example of its usage can be found in the gen_flight_plan script.

### `stats.py` `utils.py`
Extra common code used by different scripts. Do not remove these if you want the scripts to work

## Example usage

>run python3 gen_flight_plan.py
>run python3 association.py

