"""Evaluate cost of system design with 'practical' controller."""

import os
import csv
import time
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Dict, Mapping, Tuple, Union

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel



def evaluate_system(
        schema_path: Union[str, Path],
        pricing_dict: Dict[str,float],
        opex_factor: float,
        clip_level: str = 'b',
        design: bool = True,
        tau: int = 48,
        suppress_output = False
    ) -> Dict[str,Any]:
    """Evaluate performance of system defined by schema with given
    costs and control characteristics.

    Args:
        schema_path (Union[str, Path]): Path to schema defining system setup.
        pricing_dict (Dict[str,float]): Dictionary of pricing data to use.
            Keys [units]; 'carbon' [$/kgCO2], 'battery' [$/kWh], 'solar' [$/kWp]
        opex_factor (float): operational lifetime to consider OPEX costs over
            as factor of time duration considered of simulation.
        clip_level (str, optional): str, either 'd' (district) or 'b' (building),
            indicating the level at which to clip cost values in the objective
            function. Defaults to 'b'.
        design (bool, optional): Whether to include asset costs & use extended
            OPEX in reported costs. Defaults to True.
        tau (int, optional): Planning horizon for controller. Defaults to 48.
        suppress_output (bool, optional): Wether to disable outputs to
            terminal. Defaults to False

    Returns:
        Dict[str,Any]: Dictionary of system cost (objective) & breakdown.
    """

    # Initialise CityLearn environment object.
    env: CityLearnEnv = CityLearnEnv(schema=schema_path)

    # Initialise Linear MPC object.
    lp = LinProgModel(env=env)
    lp.tau = tau
    lp.generate_LP(clip_level=clip_level,pricing_dict=pricing_dict,opex_factor=opex_factor)

    # Initialise control loop.
    lp_solver_time_elapsed = 0
    num_steps = 0
    done = False

    # Initialise environment.
    observations = env.reset()
    soc_obs_index = 22
    current_socs = np.array([[charge*capacity for charge,capacity in zip(np.array(observations)[:,soc_obs_index],lp.battery_capacities)]]) # get initial SoCs

    # Execute control loop.
    with tqdm(total=env.time_steps, disable=suppress_output) as pbar:

        while not done:
            if num_steps%100 == 0:
                pbar.update(100)

            # Compute MPC action.
            # ====================================================================
            if num_steps <= (env.time_steps - 1) - tau:
                # setup and solve predictive Linear Program model of system
                lp_start = time.perf_counter()
                lp.set_time_data_from_envs(t_start=num_steps, tau=tau, current_socs=current_socs) # load ground truth data
                lp.set_LP_parameters()
                results = lp.solve_LP(ignore_dpp=False)
                actions: np.array = results['battery_inflows'][0][:,0].reshape((lp.N,1))/lp.battery_capacities
                lp_solver_time_elapsed += time.perf_counter() - lp_start

            else: # if not enough time left to grab a full length ground truth forecast: do nothing
                actions = np.zeros((lp.N,1))

            # Apply action to environment.
            # ====================================================================
            observations, _, done, _ = env.step(actions)

            # Update battery states-of-charge
            # ====================================================================
            current_socs = np.array([[charge*capacity for charge,capacity in zip(np.array(observations)[:,soc_obs_index],lp.battery_capacities)]])

            num_steps += 1

    if not suppress_output: print("Evaluation complete.")

    # Calculate objective fn performance.
    objective_contributions = []
    if clip_level == 'd':
        objective_contributions += [np.clip(np.sum([b.net_electricity_consumption for b in env.buildings],axis=0),0,None)\
                                    @ env.buildings[0].pricing.electricity_pricing]
        objective_contributions += [np.clip(np.sum([b.net_electricity_consumption for b in env.buildings],axis=0),0,None)\
                                    @ env.buildings[0].carbon_intensity.carbon_intensity * lp.pricing_dict['carbon']]
    elif clip_level == 'b':
        objective_contributions += [np.sum([np.clip(net_elec,0,None) @ env.buildings[0].pricing.electricity_pricing\
                                        for net_elec in [b.net_electricity_consumption for b in env.buildings]])]
        objective_contributions += [np.sum([np.clip(net_elec,0,None) @ env.buildings[0].carbon_intensity.carbon_intensity\
                                        for net_elec in [b.net_electricity_consumption for b in env.buildings]])\
                                            * lp.pricing_dict['carbon']]
    if design:
        objective_contributions = [contr*opex_factor for contr in objective_contributions] # extend opex costs to design lifetime
        objective_contributions += [np.sum([b.electrical_storage.capacity_history[0] for b in env.buildings]) * lp.pricing_dict['battery']]
        objective_contributions += [np.sum([b.pv.nominal_power for b in env.buildings]) * lp.pricing_dict['solar']]

    return {'objective': np.sum(objective_contributions), 'objective_contrs': objective_contributions}


if __name__ == '__main__':

    # Define test dataset.
    dataset_dir = os.path.join('A37_example_test') # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    # Define cost parameters.
    opex_factor = 10
    pricing_dict = {'carbon':5e-2,'battery':1e3,'solar':2e3}

    # Run system evaulation.
    results = evaluate_system(schema_path,pricing_dict,opex_factor)
    print(results)