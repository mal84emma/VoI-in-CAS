"""Design system using LP & quantify over-optimism."""

import os
import csv
import time
import numpy as np

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel
from schema_builder import build_schema
from sys_eval import evaluate_system

if __name__ == '__main__':

    # Set up evaluation params.
    dataset_dir = os.path.join('A37_example_validate') # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema_temp.json')
    opex_factor = 10
    pricing_dict = {'carbon':5e-1,'battery':1e3,'solar':2e3}


    # Design system using LP model.
    # ============================================================================

    # Set up base parameters of system.
    ids = [5,11,14,16,24,29]

    base_kwargs = {
        'output_dir_path': os.path.join('data','A37_example_validate'),
        'building_names': ['UCam_Building_%s'%id for id in ids],
        'battery_energy_capacities': None,
        'battery_power_capacities': [391.0,342.0,343.0,306.0,598.0,571.0], # from Annex 37
        'battery_efficiencies': [0.90]*len(ids), # from Annex 37
        'pv_power_capacities': None,
        'load_data_paths': ['UCam_Building_%s.csv'%id for id in ids],
        'weather_data_path': 'weather.csv',
        'carbon_intensity_data_path': 'carbon_intensity.csv',
        'pricing_data_path': 'pricing.csv',
        'schema_name': 'schema_temp'
    }

    # Setup initial guess.
    current_battery_capacities = [3127.0,2736.0,2746.0,2448.0,4788.0,4565.0]
    current_solar_capacities = [178.0,41.0,57.0,120.0,1349.0,257.0]

    nits = 0
    max_its = 10

    # Iterate LP solution to overcome solar mode shape assumption.
    # (this is not a huge issue it turns out, non-linearities are only for small capacities)
    while nits < max_its:
        print(f"\nSolution attempt {nits}.")

        base_kwargs.update({
            'battery_energy_capacities': current_battery_capacities,
            'pv_power_capacities': current_solar_capacities
        })
        schema_path = build_schema(**base_kwargs)

        # Initialise CityLearn environment object.
        env = CityLearnEnv(schema=schema_path)

        # Initialise Linear MPC object.
        lp = LinProgModel(env=env)
        lp.set_time_data_from_envs()
        lp.generate_LP(clip_level='b',design=True,pricing_dict=pricing_dict,opex_factor=opex_factor)
        lp.set_LP_parameters()
        lp_results = lp.solve_LP(verbose=True,ignore_dpp=True)

        print(lp_results['objective'])
        print(lp_results['objective_contrs'])
        print(lp_results['battery_capacities'])
        print(lp_results['solar_capacities'])

        if (np.all(np.abs(lp_results['battery_capacities'] - current_battery_capacities) < 0.1)) &\
            (np.all(np.abs(lp_results['solar_capacities'] - current_solar_capacities) < 0.1)):
            break
        else:
            current_battery_capacities = lp_results['battery_capacities']
            current_solar_capacities = lp_results['solar_capacities']

        nits += 1

    print('\nLP Design Complete.')
    print('===================')
    print(lp_results['objective'])
    print(lp_results['objective_contrs'])
    print(lp_results['battery_capacities'])
    print(lp_results['solar_capacities'])
    print('\n')

    # Evaluate true cost of LP designed system with real controller.
    # ============================================================================

    eval_results = evaluate_system(schema_path,pricing_dict,opex_factor)

    print('\nTrue system performance.')
    print('========================')
    print(eval_results['objective'])
    print(eval_results['objective_contrs'])

    # Report LP over-optimism.
    # ============================================================================
    print('\nLP Over-Optimism.')
    print('=================')
    print(f"{round((np.abs(eval_results['objective']-lp_results['objective'])/eval_results['objective'])*100,2)}%")


    # TODO: Repeat analysis using real predictor - e.g. Pat's linear networks
    # ============================================================================


    # TODO: Repeat analysis under stochastic setting - both stochastic LP and true expected system performance
    # ============================================================================
    # Do this in a separate script?
