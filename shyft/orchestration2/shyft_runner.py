"""
Main module for running an SHyFT simulation.

This script expects a configuration files as a parameter where there is
supposed to be info on where other region, model, datasets and other
parameter config files are to be found.

More help on supported parameters by typing 'name_of_this_script -h'.
"""

from __future__ import print_function
from __future__ import absolute_import

import argparse

from shyft import api
from shyft.orchestration2 import (
    config_constructor, CalibrationConfig, Simulator, Calibrator, BaseSimulationOutput, get_class)


def make_fake_target(config, time_axis, catchment_index):
    print("Fake target Catchment index = {}".format(catchment_index))
    tv = api.TargetSpecificationVector()
    t = api.TargetSpecificationPts()
    simulator = Simulator(config)
    simulator.build_model(time_axis.start(), time_axis.delta(), time_axis.size())
    print("fake target run the model")
    simulator.run_model()
    print("fake target get the results out")

    ts = simulator.get_sum_catchment_result('SimDischarge', catchment_index)
    mapped_indx = [i for i, j in enumerate(simulator.catchment_map) if j in catchment_index]
    catch_indx = api.IntVector(mapped_indx)
    t.catchment_indexes = catch_indx
    t.scale_factor = 1.0
    t.calc_mode = api.NASH_SUTCLIFFE
    t.ts = ts
    tv.push_back(t)
    return tv


def main_calibration_runner(config_file, section):
    print('Starting calibration runner')
    config = CalibrationConfig(config_file, section)
    time_axis = api.Timeaxis(config.model_config.start_time, config.model_config.run_time_step,
                                          config.model_config.number_of_steps)
    #config._target = make_fake_target(config.model_config, time_axis, config.catchment_index[0]['catch_id'])
    calibrator = Calibrator(config)
    calibrator.init(time_axis)
    calibr_results = calibrator.calibrate(tol=1.0e-5)
    print("Calibration result:", calibr_results)
    if hasattr(config, "calibrated_model"):
        calibrator.save_calibrated_model(config.calibrated_model, calibr_results)
    return calibrator



# Save the output in a function specified in:
# ----
# output:
#   class: ...
#   file: ...
#   params:
#     ...
# ...
def save_output(simulator, config):
    """Save selected output from simulator in a netcdf file"""
    # Get the handles class for output
    out_class = get_class(config.output['class'])
    cells = simulator.model.get_cells()
    out_instance = out_class(config.output['params'])
    if not isinstance(out_instance, BaseSimulationOutput):
        raise ValueError("The repository class is not an instance of 'BaseSimulationOutput'")
    outfile = config.abspath(config.output['file'])
    out_instance.save_output(cells, outfile)


# Entry point for runner script
def main():
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", "-c",
                            help="The main SHyFT run config file.",
                            required=True)
    arg_parser.add_argument("--config_section", "-s",
                            help="The section inside the main SHyFT run config file.",
                            required=True)
    arg_parser.add_argument("--calibration", "-l", action="store_true",
                            help="Config file is a calibration one.",
                            required=False)
    args = arg_parser.parse_args()

    if args.calibration:
        # Just a calibration run
        return main_calibration_runner(args.config, args.config_section)

    # Get the configuration section
    run_config = config_constructor(args.config, args.config_section)

    # Build the simulator
    simulator = Simulator(run_config)
    time_axis = api.Timeaxis(run_config.start_time, run_config.run_time_step, run_config.number_of_steps)
    simulator.build_model(time_axis.start(), time_axis.delta(), time_axis.size())
    simulator.run_model()

    # Get output out of the simulator
    if hasattr(run_config, "output"):
        save_output(simulator, run_config)
    return simulator



if __name__ == "__main__":
    main()
