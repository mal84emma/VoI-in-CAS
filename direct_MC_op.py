"""Directly optimise MC estimate of expected utility to solve prior problem."""

import os
import json
import time
import numpy as np
import scipy.optimize as op

from mproc_utils import parallel_task, multi_proc_constr_and_eval_system

def cost_MC_estimate(x, ids, n_samples):

    print(x)

    assert len(x) % 2 == 0, "Design variable argument must have even length."

    # Define common parameters (this is hacky, think about changing)
    # ========================================================================
    dataset_dir = os.path.join('data','A37_analysis_test')
    opex_factor = 10
    pricing_dict = {'carbon':5e-1,'battery':1e3,'solar':2e3}
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

    # Grab system design parameters for evaluation.
    battery_energy_capacities = x[:len(x)//2]
    pv_power_capacities = x[len(x)//2:]

    # Old serial implementation.
    # costs = []

    # for _ in range(n_samples):

    #     # Make draw from distribution of uncertainties (also hacky)
    #     # ========================================================================
    #     mu = 0.85
    #     sigma = 0.1
    #     eta_samples = np.random.normal(loc=mu,scale=sigma,size=(len(ids)))
    #     eta_samples = np.clip(eta_samples,0,1)

    #     # Construct schema with specified decision varaibles and samples parameters
    #     # ========================================================================
    #     base_kwargs.update({
    #             'battery_energy_capacities': battery_energy_capacities,
    #             'battery_efficiencies': eta_samples,
    #             'pv_power_capacities': pv_power_capacities
    #         })
    #     schema_path = build_schema(**base_kwargs)


    #     # Evaluate system cost
    #     # ========================================================================
    #     eval_results = evaluate_system(schema_path,pricing_dict,opex_factor)
    #     cost = eval_results['objective']
    #     costs.append(cost)


    # Compute MC estimate of system cost for design. (Parallelised)
    # ============================================================================

    # Draw from distribution of uncertain parameters.
    mu = 0.85
    sigma = 0.1
    eta_samples = np.random.normal(loc=mu,scale=sigma,size=(n_samples,len(ids)))
    eta_samples = np.clip(eta_samples,0,1)

    # Evaluate system cost for each parameter value in parallel.
    n_processes = min(25,os.cpu_count()//2)

    mproc_args_list = [[battery_energy_capacities,pv_power_capacities,eta_samples[n],base_kwargs,pricing_dict,opex_factor,n] for n in range(n_samples)]
    costs = parallel_task(multi_proc_constr_and_eval_system, mproc_args_list, n_procs=n_processes)

    return np.mean(costs)


if __name__ == '__main__':

    seed = 42
    ids = [48]
    n_samples = 50
    lower_bounds = np.array([*[5e2]*len(ids),*[1e2]*len(ids)])
    upper_bounds = np.array([*[2e3]*len(ids),*[1.5e3]*len(ids)])
    bounds = op.Bounds(lb=lower_bounds,ub=upper_bounds)
    #results = op.dual_annealing(cost_MC_estimate, bounds, args=(ids,n_samples), seed=seed)
    start = time.time()
    results = op.differential_evolution(cost_MC_estimate, bounds, args=(ids,n_samples), seed=seed)
    end = time.time()

    print(results.x, results.fun, results.message)
    print("Runtime: %s s" % round(end-start,1))
    print("No. of fn evals: %s"%results.nfev)

# RESULTS
# =======
#
# ...
# ... 4944989.411288872 Optimization terminated successfully.
# Runtime: ... s
# No. of fn evals: ... @ 50 samples per MC estimate eval (25 processes mproc)
