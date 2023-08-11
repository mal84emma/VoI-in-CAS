"""Compute cost of building operation without solar-battery assets."""

import os
import csv
import json
import time
import numpy as np

from citylearn.citylearn import CityLearnEnv
from schema_builder import build_schema
from sys_eval import construct_and_evaluate_system



if __name__ == '__main__':

    # Set up evaluation params.
    dataset_dir = os.path.join('data','A37_analysis_test') # dataset directory
    opex_factor = 10
    pricing_dict = {'carbon':5e-1,'battery':1e3,'solar':2e3}


    # Evaluate performance of system without solar & batteries.
    # ============================================================================

    # Set up base parameters of system.
    #ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49]
    b_ids = [48,32,16,0,25,15,3,11,12,49]

    results = {}

    for B in [1]:

        ids = b_ids[:B]

        battery_efficiencies = [0.85]*len(ids) # 0.9 for Annex 37
        with open(os.path.join(dataset_dir,'metadata_ext.json'),'r') as json_file:
            annex_defaults = json.load(json_file)

        base_kwargs = {
            'output_dir_path': dataset_dir,
            'building_names': ['UCam_Building_%s'%id for id in ids],
            'battery_energy_capacities': None,
            'battery_power_capacities': [annex_defaults["building_attributes"]["battery_power_capacities (kW)"][str(id)] for id in ids],
            'battery_efficiencies': battery_efficiencies,
            'pv_power_capacities': None,
            'load_data_paths': ['UCam_Building_%s.csv'%id for id in ids],
            'weather_data_path': 'weather.csv',
            'carbon_intensity_data_path': 'carbon_intensity.csv',
            'pricing_data_path': 'pricing.csv',
            'schema_name': 'schema_temp'
        }

        # Evaluate true cost of LP designed system with real controller.
        # ============================================================================
        eval_results = construct_and_evaluate_system(
                [1e-9]*len(ids), [1e-9]*len(ids), battery_efficiencies,
                base_kwargs, pricing_dict, opex_factor,
                return_contrs=True, suppress_output=False,
                no_control=True
            )

        print('\Original system performance.')
        print('========================')
        print(eval_results['objective'])
        print(eval_results['objective_contrs'])

# Results
# =======
#
# 11/8/23
# 13331208.157186778
# [10313780.053687025, 3017428.1034967536, 1.0000000000000002e-06, 2.0000000000000003e-06]