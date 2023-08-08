"""Design system using stochastic LP & quantify over-optimism."""

import os
import csv
import json
import time
import numpy as np

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel
from schema_builder import build_schema
from sys_eval import construct_and_evaluate_system
from mproc_utils import parallel_task, multi_proc_constr_and_eval_system



if __name__ == '__main__':

    # Set up evaluation params.
    dataset_dir = os.path.join('data','A37_analysis_test') # dataset directory
    opex_factor = 10
    pricing_dict = {'carbon':5e-1,'battery':1e3,'solar':2e3}

    # Design system using LP scenario optimisation model.
    # ============================================================================

    # Set up base parameters of system.
    #ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49]
    ids = [48]

    with open(os.path.join(dataset_dir,'metadata_ext.json'),'r') as json_file:
        annex_defaults = json.load(json_file)

    base_kwargs = {
        'output_dir_path': dataset_dir,
        'building_names': ['UCam_Building_%s'%id for id in ids],
        'battery_energy_capacities': None,
        'battery_power_capacities': [annex_defaults["building_attributes"]["battery_power_capacities (kW)"][str(id)] for id in ids],
        'battery_efficiencies': None,
        'pv_power_capacities': None,
        'load_data_paths': ['UCam_Building_%s.csv'%id for id in ids],
        'weather_data_path': 'weather.csv',
        'carbon_intensity_data_path': 'carbon_intensity.csv',
        'pricing_data_path': 'pricing.csv',
        'schema_name': 'schema_temp'
    }

    # Setup initial guess - from Annex 37.
    current_battery_capacities = [annex_defaults["building_attributes"]["battery_energy_capacities (kWh)"][str(id)] for id in ids]
    current_solar_capacities = [annex_defaults["building_attributes"]["pv_power_capacities (kW)"][str(id)] for id in ids]

    base_kwargs.update({
        'battery_energy_capacities': current_battery_capacities,
        'pv_power_capacities': current_solar_capacities
    })

    # Set up probabilistic model of effiencies and take draws.
    n_draws = 100
    mu = 0.85
    sigma = 0.1
    eta_samples = np.random.normal(loc=mu,scale=sigma,size=(n_draws,len(ids)))
    eta_samples = np.clip(eta_samples,0,1)

    # Set up scenario optimisation object.
    num_scenarios = 3

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
            lp = LinProgModel(env=envs[m])
        else:
            lp.add_env(env=envs[m])

    lp.set_time_data_from_envs()
    lp.generate_LP(clip_level='m',design=True,pricing_dict=pricing_dict,opex_factor=opex_factor)
    lp.set_LP_parameters()
    lp_results = lp.solve_LP(verbose=True,ignore_dpp=True)

    print('\nLP Stochastic Design Complete.')
    print('===================')
    print(lp_results['objective'])
    print(lp_results['objective_contrs'])
    print(lp_results['battery_capacities'])
    print(lp_results['solar_capacities'])
    print('\n')


    # Compute MC estimate of true system cost for design
    # ============================================================================
    n_samples = 100
    n_processes = min(25,os.cpu_count()//2)

    mproc_args_list = [[lp_results['battery_capacities'],lp_results['solar_capacities'],eta_samples[n],base_kwargs,pricing_dict,opex_factor,n] for n in range(n_samples)]
    cost_evals = parallel_task(multi_proc_constr_and_eval_system, mproc_args_list, n_procs=n_processes)


    # Report stochastic LP over-optimism.
    # ============================================================================
    print('\nStochastic LP Over-Optimism.')
    print('=================')
    print(f"{round((np.abs(np.mean(cost_evals)-lp_results['objective'])/np.mean(cost_evals))*100,2)}%")


    # Repeat analysis using real controller?
    # ============================================================================
