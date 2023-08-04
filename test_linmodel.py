"""Test functioning of `linmodel` LP solving."""

import os
import csv
import time
import numpy as np

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel


if __name__ == '__main__':

    dataset_dir = os.path.join('A37_example_test') # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')
    #tau = 24*7*52 # model prediction horizon (number of timesteps of data predicted)
    #opex_factor = (24*7*52*20)/tau
    opex_factor = 10

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Linear MPC object.
    lp = LinProgModel(env=env)
    #for i in range(2): lp.add_env(env=env)
    #lp.tau = tau
    lp.set_time_data_from_envs()
    lp.generate_LP(clip_level='m',design=True,pricing_dict={'carbon':5e-1,'battery':1e3,'solar':2e3},opex_factor=opex_factor)
    lp.set_LP_parameters()
    results = lp.solve_LP(verbose=True,ignore_dpp=True)

    print(results['objective'])
    print(results['objective_contrs'])
    print(results['battery_capacities'])
    print(results['solar_capacities'])