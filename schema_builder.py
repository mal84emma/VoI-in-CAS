"""Build schema for a specified system design/realisation."""

import os
import csv
import json
from pathlib import Path
from typing import Any, List, Dict, Mapping, Tuple, Union


def build_schema(
        output_dir_path: Union[str, Path],
        building_names: List[str],
        battery_energy_capacities: List[float],
        battery_power_capacities: List[float],
        battery_efficiencies: List[float],
        pv_power_capacities: List[float],
        load_data_paths: List[Union[str, Path]],
        weather_data_path: Union[str, Path],
        carbon_intensity_data_path: Union[str, Path],
        pricing_data_path: Union[str, Path],
        simulation_duration: int = None,
        schema_name: str = 'schema_temp'
    ) -> Union[str, Path]:
    """Construct a schema.json for the specified system design/realisation
    and save to file.

    Note: all data file paths must be specified relative to the schema
    location, so the schema must be saved to the dir containing the data.

    Args:
        output_dir_path (Union[str, Path]): Path to output directory
            (directory containing data files).
        building_names (List[str]): List of building names in system.
        battery_energy_capacities (List[float]): List of energy capacities
            of batteries in system.
        battery_power_capacities (List[float]): List of power capacities
            of batteries in system.
        battery_efficiencies (List[float]): List of (round-trip)
            efficiencies of batteries in system.
        pv_power_capacities (List[float]): List of solar panel power
            capacities in system.
        load_data_paths (List[Union[str, Path]]): List of paths to
            building data CSV for each building.
        weather_data_path (Union[str, Path]): Path to weather data CSV.
        carbon_intensity_data_path (Union[str, Path]): Path to carbon
            intensity data CSV.
        pricing_data_path (Union[str, Path]): Path to pricing data CSV.
        simulation_duration (int, optional): Number of data time instances
            from simulation data to be used. Max is total number of data
            points in CSVs. Defaults to None.
        schema_name (str): Name to be used for output schema file. Defaults
            to schema_temp.

    Returns:
        Union[str, Path]: Full path to created schema file.
    """

    # load base schema
    with open(os.path.join('resources','base_schema.json')) as base_schema:
        schema = json.load(base_schema)

    if simulation_duration is None:
        with open(os.path.join(output_dir_path,pricing_data_path)) as file:
            reader_file = csv.reader(file)
            simulation_duration = len(list(reader_file))-1 # skip header row

    schema['simulation_end_time_step'] = simulation_duration - 1 # set length of simulation

    # write building attributes
    schema['buildings'] = {}
    for i,b_name in enumerate(building_names):

        building_dict = {
            'include': True,
            'energy_simulation': load_data_paths[i],
            'weather': weather_data_path,
            'carbon_intensity': carbon_intensity_data_path,
            'pricing': pricing_data_path,
            'inactive_observations': [],
            'inactive_actions': [],

            'electrical_storage': {
                'type': "citylearn.energy_model.Battery",
                'autosize': False,
                'attributes': {
                        'capacity': battery_energy_capacities[i],
                        'nominal_power': battery_power_capacities[i],
                        'capacity_loss_coefficient': 1e-05,
                        'loss_coefficient': 0,
                        'power_efficiency_curve': [[0,battery_efficiencies[i]],[1,battery_efficiencies[i]]],
                        'capacity_power_curve': [[0,1],[1,1]]
                }
            },

            'pv': {
                'type': "citylearn.energy_model.PV",
                'autosize': False,
                'attributes': {'nominal_power': pv_power_capacities[i]}
            }
        }

        schema['buildings'].update({b_name: building_dict})

    # write schema to file
    schema_path = os.path.join(output_dir_path,'%s.json'%schema_name)
    with open(schema_path, 'w') as file:
        json.dump(schema, file, indent=4)

    return schema_path


if __name__ == '__main__':

    ids = [5,11,14]

    base_kwargs = {
        'output_dir_path': os.path.join('data','A37_example_validate'),
        'building_names': ['UCam_Building_%s'%id for id in ids],
        'battery_energy_capacities': [2000,1500,1500],
        'battery_power_capacities': [300,250,250],
        'battery_efficiencies': [0.95,0.95,0.95],
        'pv_power_capacities': [150,50,50],
        'load_data_paths': ['UCam_Building_%s.csv'%id for id in ids],
        'weather_data_path': 'weather.csv',
        'carbon_intensity_data_path': 'carbon_intensity.csv',
        'pricing_data_path': 'pricing.csv',
        'schema_name': 'schema_build_test'
    }

    schema_path = build_schema(**base_kwargs)