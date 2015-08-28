# -*- coding: utf-8 -*-

import sys
#sys.path.insert(0, "D:/Users/ysa/projects")

from shyft.orchestration.utils.EnkiRunner import EnkiRunner
from shyft.orchestration.utils.EnkiConfig import EnkiConfig
import os

# Step #1: Figure out where we have the runner_configurations yaml file
#enki_root=os.path.join("D:/","enki")
enki_root=os.path.join("D:/","Users/ysa")
config_file = os.path.join(enki_root, "config","runner_configurations.yaml")
# Step #2: Feed the config file to the EnkiConfig class, with a configuration that we would like to run
#          Note that on the next line, EnkiConfig will read all needed info
#            that is: 
#                1. cells (from catchment ids -and region bounding box)
#                2. timeseries (from station id and smg mapping)
#                3. cell-states (from the states directory)
#config= EnkiConfig(config_file,'NeaNidelva')
#config= EnkiConfig(config_file,'JostedalLeirdola')
#config= EnkiConfig(config_file,'Tokke')
#config= EnkiConfig(config_file,'Rana')
config= EnkiConfig(config_file,'Vik')
# Step #3: Feed the loaded configuration to the EnkiRunner, 
#          build_model for the time-axis (interpolation step), and run..
simulator = EnkiRunner(config)
simulator.build_model(config.t_start, config.dt, config.n_steps)
simulator.run_model()
if "shapes" in simulator.cell_data_types():
    extractor = {'Total discharge': lambda x: x.response.total_discharge,
#                 'Snow storage': lambda x: x.response.gs.storage*(1.0 - (x.lake_frac + x.reservoir_frac)),
                 'Relative_humidity': lambda x: x.rel_hum[0],
                 'Temperature': lambda x: x.temperature[len(x.temperature)-1]}#,
#                 'Precipitation': lambda x: x.precipitation[len(x.precipitation)-1]}
    simulator.plot_distributed_data(simulator.cell_data("shapes"), simulator.model.get_cells(), extractor)
