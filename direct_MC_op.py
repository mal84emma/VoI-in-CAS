"""Directly optimise MC estimate of expected utility to solve prior problem."""

import os
import time
import numpy as np
import scipy.optimize as op

from sys_eval import evaluate_system
from schema_builder import build_schema

def cost_MC_estimate(x, ids, n_samples):

    print(x)

    assert len(x) % 2 == 0, "Design variable argument must have even length."

    # Define common parameters (this is hacky, think about changing)
    # ========================================================================
    opex_factor = 10
    pricing_dict = {'carbon':5e-1,'battery':1e3,'solar':2e3}
    base_kwargs = {
        'output_dir_path': os.path.join('data','A37_example_validate'),
        'building_names': ['UCam_Building_%s'%id for id in ids],
        'battery_energy_capacities': None,
        'battery_power_capacities': [391.0,342.0,343.0,306.0,598.0,571.0], # from Annex 37
        'battery_efficiencies': None,
        'pv_power_capacities': None,
        'load_data_paths': ['UCam_Building_%s.csv'%id for id in ids],
        'weather_data_path': 'weather.csv',
        'carbon_intensity_data_path': 'carbon_intensity.csv',
        'pricing_data_path': 'pricing.csv',
        'schema_name': 'schema_temp'
    }

    battery_energy_capacities = x[:len(x)//2]
    pv_power_capacities = x[len(x)//2:]

    costs = []

    for _ in range(n_samples):

        # Make draw from distribution of uncertainties (also hacky)
        # ========================================================================
        mu = 0.85
        sigma = 0.1
        eta_samples = np.random.normal(loc=mu,scale=sigma,size=(len(ids)))
        eta_samples.clip(0,1)

        # Construct schema with specified decision varaibles and samples parameters
        # ========================================================================
        base_kwargs.update({
                'battery_energy_capacities': battery_energy_capacities,
                'battery_efficiencies': eta_samples,
                'pv_power_capacities': pv_power_capacities
            })
        schema_path = build_schema(**base_kwargs)


        # Evaluate system cost
        # ========================================================================
        eval_results = evaluate_system(schema_path,pricing_dict,opex_factor)
        cost = eval_results['objective']
        costs.append(cost)

    return np.mean(costs)


if __name__ == '__main__':

    seed = 42
    ids = [11]
    n_samples = 1
    upper_bounds = np.array([*[5e4]*len(ids),*[5e3]*len(ids)])
    bounds = op.Bounds(lb=np.ones(len(ids)*2),ub=upper_bounds)
    #results = op.dual_annealing(cost_MC_estimate, bounds, args=(ids,n_samples), seed=seed)
    start = time.time()
    results = op.differential_evolution(cost_MC_estimate, bounds, args=(ids,n_samples), seed=seed)
    end = time.time()

    print(results.x, results.fun, results.message)
    print("Runtime: %s s" % round(end-start,1))