"""
Module for reading region netCDF files needed for an SHyFT run.
"""

from __future__ import absolute_import

import numpy as np
from netCDF4 import Dataset

from ..base_config import BaseSimulationOutput
from ..utils import cell_extractor


class SimulationOutput(BaseSimulationOutput):

    def __init__(self, params):
        self.params = params

    def save_output(self, cells, outfile):

        print("Saving simulation output in file:", outfile)
        with Dataset(outfile, "w") as d:
            for i, cell in enumerate(cells):
                # print("cell:", i)
                grp = d.createGroup('cell%d' % i)
                grp.createDimension('time', None)
                for param in self.params:
                    value = cell_extractor[param](cell)
                    if type(value) in (list, np.ndarray):
                        # print("    %s: [%s, ...]" % (param, value[0]))
                        nc_param = grp.createVariable(
                            param, 'f8', ('time',), zlib=True, shuffle=True, least_significant_digit=3)
                        nc_param[:] = value
                    else:
                        print("    %s: %s" % (param, value))
                        setattr(grp, param, value)
