"""Compute VSS for LP design methods."""

import os
import csv
import copy
import json
import time
import numpy as np
from collections.abc import Iterable

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel
from schema_builder import build_schema



if __name__ == '__main__':

    # Set up evaluation params.
    dataset_dir = os.path.join('data','A37_analysis_test') # dataset directory
    opex_factor = 10
    pricing_dict = {'carbon':5e-1,'battery':1e3,'solar':2e3}

    # Design system using LP scenario optimisation model.
    # ============================================================================

    # Set up base parameters of system.
    #b_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49]
    b_ids = [48,32,16]

    with open(os.path.join(dataset_dir,'metadata_ext.json'),'r') as json_file:
        annex_defaults = json.load(json_file)

    results = {}

    for B,N in zip([1,2,3],[25,15,8]):

        print(B, N, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        ids = b_ids[:B]

        # Setup system attributes
        # ========================================================================
        base_kwargs = {
            'output_dir_path': dataset_dir,
            'building_names': ['UCam_Building_%s'%id for id in ids],
            'battery_energy_capacities': [annex_defaults["building_attributes"]["battery_energy_capacities (kWh)"][str(id)] for id in ids],
            'battery_power_capacities': [annex_defaults["building_attributes"]["battery_power_capacities (kW)"][str(id)] for id in ids],
            'battery_efficiencies': [0.85]*len(ids),
            'pv_power_capacities': [annex_defaults["building_attributes"]["pv_power_capacities (kW)"][str(id)] for id in ids],
            'load_data_paths': ['UCam_Building_%s.csv'%id for id in ids],
            'weather_data_path': 'weather.csv',
            'carbon_intensity_data_path': 'carbon_intensity.csv',
            'pricing_data_path': 'pricing.csv',
            'schema_name': 'schema_temp'
        }

        # Solve deterministic LP.
        # ========================================================================
        schema_path = build_schema(**base_kwargs)

        # Initialise CityLearn environment object.
        env = CityLearnEnv(schema=schema_path)

        # Initialise Linear MPC object.
        det_lp = LinProgModel(env=env)
        det_lp.set_time_data_from_envs()
        det_lp.generate_LP(clip_level='m',design=True,pricing_dict=pricing_dict,opex_factor=opex_factor)
        det_lp.set_LP_parameters()
        det_lp_results = det_lp.solve_LP(verbose=True,ignore_dpp=True)

        print('\nDeterministic LP Design Complete.')
        print('===================')
        print(det_lp_results['objective'])
        print(det_lp_results['objective_contrs'])
        print(det_lp_results['battery_capacities'])
        print(det_lp_results['solar_capacities'])
        print('\n')


        # Do sampling.
        # ========================================================================

        # Set up probabilistic model of effiencies and take draws.
        n_draws = 100
        mu = 0.85
        sigma = 0.1
        eta_samples = np.random.normal(loc=mu,scale=sigma,size=(n_draws,len(ids)))
        eta_samples = np.clip(eta_samples,0,1)

        num_scenarios = N


        # Solve stochastic LP.
        # ========================================================================
        envs = []

        for m in range(num_scenarios):

            # Build schema.
            base_kwargs.update({
                    'battery_efficiencies': eta_samples[m]
                })
            schema_path = build_schema(**base_kwargs)

            # Initialise CityLearn environment object.
            envs.append(CityLearnEnv(schema=schema_path))

            if m == 0: # initialise lp object
                stoch_lp = LinProgModel(env=envs[m])
            else:
                stoch_lp.add_env(env=envs[m])

        stoch_lp.set_time_data_from_envs()
        stoch_lp.generate_LP(clip_level='m',design=True,pricing_dict=pricing_dict,opex_factor=opex_factor)
        stoch_lp.set_LP_parameters()
        stoch_lp_results = stoch_lp.solve_LP(verbose=True,ignore_dpp=True)

        print('\nStochastic LP Design Complete.')
        print('===================')
        print(stoch_lp_results['objective'])
        print(stoch_lp_results['objective_contrs'])
        print(stoch_lp_results['battery_capacities'])
        print(stoch_lp_results['solar_capacities'])
        print('\n')


        # Compute stochastic performance of deterministic LP solution.
        # ========================================================================
        envs = []

        for m in range(num_scenarios):

            # Build schema.
            base_kwargs.update({
                    'battery_energy_capacities': det_lp_results['battery_capacities'],
                    'pv_power_capacities': det_lp_results['solar_capacities'],
                    'battery_efficiencies': eta_samples[m]
                })
            schema_path = build_schema(**base_kwargs)

            # Initialise CityLearn environment object.
            envs.append(CityLearnEnv(schema=schema_path))

            if m == 0: # initialise lp object
                comp_lp = LinProgModel(env=envs[m])
            else:
                comp_lp.add_env(env=envs[m])

        comp_lp.set_time_data_from_envs()
        comp_lp.generate_LP(clip_level='m',design=False,pricing_dict=pricing_dict,opex_factor=opex_factor)
        comp_lp.set_LP_parameters()
        comp_lp_results = comp_lp.solve_LP(verbose=True,ignore_dpp=True)

        print('\nDet. LP Solution Checking Complete.')
        print('===================')
        print(comp_lp_results['objective'])
        print(comp_lp_results['objective_contrs'])
        print(comp_lp_results['battery_capacities'])
        print(comp_lp_results['solar_capacities'])
        print('\n')


        # Store results.
        # ========================================================================
        keys = ['objective','objective_contrs','battery_capacities','solar_capacities']
        results[B] = {
            'det':{key:list(det_lp_results[key]) if isinstance(det_lp_results[key],Iterable) else det_lp_results[key] for key in keys},
            'stoch':{key:list(stoch_lp_results[key]) if isinstance(stoch_lp_results[key],Iterable) else stoch_lp_results[key] for key in keys},
            'comp':{key:list(comp_lp_results[key]) if isinstance(comp_lp_results[key],Iterable) else comp_lp_results[key] for key in keys}
        }

    print(results)

    with open('VSS_results.json','w') as json_file:
        json.dump(results,json_file, indent=4)