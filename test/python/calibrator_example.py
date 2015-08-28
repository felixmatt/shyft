# -*- coding: utf-8 -*-

import sys
#sys.path.insert(0, "D:/Users/ysa/projects")

#import copy
#import numpy as np
#import matplotlib.pyplot as plt
from shyft.orchestration.utils.EnkiRunner import EnkiRunner,make_fake_target
from shyft.orchestration.utils.EnkiRunner import EnkiCalibrator
from shyft.orchestration.utils.EnkiConfig import EnkiConfig,CalibrationConfig
from shyft import api
import os
#import sys
#import datetime as dt


enki_root=os.path.join("D:/","Users/ysa")

config_file = os.path.join(enki_root, "config","calibration_configurations.yaml")

#config= CalibrationConfig(config_file,'NeaNidelva')
config= CalibrationConfig(config_file,'Vik')

calibrator = EnkiCalibrator(config)

time_axis = api.FixedIntervalTimeAxis(config.model_config.t_start, config.model_config.dt, config.model_config.n_steps)

#config._target = make_fake_target(config.model_config, time_axis, config.catchment_index)

calibrator.init(time_axis)

opt_param=calibrator.calibrate()
print opt_param
print "1-nash =", calibrator.calculate_goal_function(opt_param)

