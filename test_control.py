"""Test `linmodel` used for controlling CityLearnEnv."""

import os
import csv
import time
import numpy as np

from tqdm import tqdm

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel


if __name__ == '__main__':

    tau = 48
    clip_level = 'b'
    opex_factor = 10
    design = False

    # Define test dataset.
    dataset_dir = os.path.join('A37_example_test') # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    # Initialise CityLearn environment object.
    env: CityLearnEnv = CityLearnEnv(schema=schema_path)

    # Initialise Linear MPC object.
    lp = LinProgModel(env=env)
    lp.tau = tau
    lp.generate_LP(clip_level=clip_level)

    # Initialise control loop.
    lp_solver_time_elapsed = 0
    num_steps = 0
    done = False

    # Initialise environment.
    observations = env.reset()
    soc_obs_index = 22
    current_socs = np.array([[charge*capacity for charge,capacity in zip(np.array(observations)[:,soc_obs_index],lp.battery_capacities)]]) # get initial SoCs

    # Execute control loop.
    with tqdm(total=env.time_steps) as pbar:

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

    print("Evaluation complete.")

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

    print(objective_contributions)
    print(np.sum(objective_contributions))