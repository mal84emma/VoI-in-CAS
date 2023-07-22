"""Wrapper function for multiprocessing system evaluations in ipynb."""

from sys_eval import evaluate_system
from schema_builder import build_schema


def construct_and_evaluate_system(battery_capacities,solar_capacities,battery_efficiencies,base_kwargs,pricing_dict,opex_factor):

    base_kwargs.update({
                'battery_energy_capacities': battery_capacities,
                'battery_efficiencies': battery_efficiencies,
                'pv_power_capacities': solar_capacities
            })
    schema_path = build_schema(**base_kwargs)

    eval_results = evaluate_system(schema_path,pricing_dict,opex_factor,suppress_output=True)

    return eval_results['objective']