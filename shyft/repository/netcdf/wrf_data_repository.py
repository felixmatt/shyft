from __future__ import absolute_import
from __future__ import print_function
from six import iteritems
from builtins import range

import os
import re
from glob import glob
from os import path
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from pyproj import transform
from shyft import api
from .. import interfaces
from .time_conversion import convert_netcdf_time


class WRFDataRepositoryError(Exception):
    pass


class WRFDataRepository(interfaces.GeoTsRepository):
    """
    Repository for geo located timeseries given as WRF(*) data in
    netCDF(3) files.

    NetCDF dataset assumptions:
        * Dimensions:
           Time = UNLIMITED ; // (1 currently)
           DateStrLen = 19 ;
           west_east = 73 ;
           south_north = 60 ;
           bottom_top = 29 ;
           bottom_top_stag = 30 ;
           soil_layers_stag = 4 ;
           west_east_stag = 74 ;
           south_north_stag = 61 ;
        * Variables:
          TODO: A lot.  We really want to list them here?

    (*) WRF model output is from:
        http://www2.mmm.ucar.edu/wrf/users/docs/user_guide_V3/users_guide_chap5.htm

    """

    _G = 9.80665 #  WMO-defined gravity constant to calculate the height in metres from geopotential

    def __init__(self, epsg, directory, filename=None, bounding_box=None,
                 x_padding=5000.0, y_padding=5000.0, allow_subset=False):
        """
        Construct the netCDF4 dataset reader for data from WRF NWP model,
        and initialize data retrieval.

        Parameters
        ----------
        epsg: string
            Unique coordinate system id for result coordinates.
            Currently "32632" and "32633" are supported.
        directory: string
            Path to directory holding one or possibly more WRF data files.
            os.path.isdir(directory) should be true, or exception is raised.
        filename: string, optional
            Name of netcdf file in directory that contains spatially
            distributed input data. Can be a glob pattern as well, in case
            it is used for forecasts or ensambles.
        bounding_box: list, optional
            A list on the form:
            [[x_ll, x_lr, x_ur, x_ul],
             [y_ll, y_lr, y_ur, y_ul]],
            describing the outer boundaries of the domain that should be
            extracted. Coordinates are given in epsg coordinate system.
        x_padding: float, optional
            Longitudinal padding in meters, added both east and west
        y_padding: float, optional
            Latitudinal padding in meters, added both north and south
        allow_subset: bool
            Allow extraction of a subset of the given source fields
            instead of raising exception.
        """
        directory = directory.replace('${SHYFTDATA}', os.getenv('SHYFTDATA', '.'))
        self._filename = path.join(directory, filename)
        self.allow_subset = allow_subset
        if not path.isdir(directory):
            raise WRFDataRepositoryError("No such directory '{}'".format(directory))

        self.shyft_cs = "+init=EPSG:{}".format(epsg)
        self._x_padding = x_padding
        self._y_padding = y_padding
        self._bounding_box = bounding_box

        # Field names and mappings
        self.wrf_shyft_map = {
            "Q2": "relative_humidity",
            "T2": "temperature",
            "HGT": "z",
            "PREC_ACC_NC": "precipitation",
            "U": "x_wind",
            "V": "y_wind",
            "SWDOWN": "radiation"}

        # Fields that need an additional timeslice because the measure average values
        self._shift_fields = ("PREC_ACC_NC", "SWDOWN")

        self.source_type_map = {"relative_humidity": api.RelHumSource,
                                "temperature": api.TemperatureSource,
                                "precipitation": api.PrecipitationSource,
                                "radiation": api.RadiationSource,
                                "wind_speed": api.WindSpeedSource}

    # TODO: We still need to adapt this to the pattern structure of WRF files
    def get_timeseries(self, input_source_types, utc_period, geo_location_criteria=None):
        """Get shyft source vectors of time series for input_source_types

        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve (precipitation, temperature..)
        geo_location_criteria: object, optional
            Some type (to be decided), extent (bbox + coord.ref)
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.

        Returns
        -------
        geo_loc_ts: dictionary
            dictionary keyed by time series name, where values are api vectors of geo
            located timeseries.
        """
        filename = self._filename
        if not path.isfile(filename):
            if '*' in filename:
                filename = self._get_files(utc_period.start, "_(\d{8})([T_])(\d{2})(Z)?.nc$")
            else:
                raise WRFDataRepositoryError("File '{}' not found".format(filename))
        with Dataset(filename) as dataset:
            return self._get_data_from_dataset(dataset, input_source_types,
                                               utc_period, geo_location_criteria)

    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria=None):
        """
        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve. Valid types are:
                * relative_humidity
                * temperature
                * precipitation
                * radiation
                * wind_speed
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.
        t_c: long
            Forecast specification; return newest forecast older than t_c.
        geo_location_criteria: object
            Some type (to be decided), extent (bbox + coord.ref).

        Returns
        -------
        geo_loc_ts: dictionary
            dictionary keyed by ts type, where values are api vectors of geo
            located timeseries.
        """
        filename = self._get_files(t_c, "_(\d{8})([T_])(\d{2})(Z)?.nc$")
        with Dataset(filename) as dataset:
            return self._get_data_from_dataset(dataset, input_source_types, utc_period,
                                               geo_location_criteria)

    def get_forecast_ensemble(self, input_source_types, utc_period,
                              t_c, geo_location_criteria=None):
        """
        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve (precipitation, temperature, ...)
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.
        t_c: long
            Forecast specification; return newest forecast older than t_c.
        geo_location_criteria: object
            Some type (to be decided), extent (bbox + coord.ref).

        Returns
        -------
        ensemble: list of geo_loc_ts dictionaries
            Dictionaries are keyed by time series type, with values
            being api vectors of geo located timeseries.
        """

        filename = self._get_files(t_c, "\D(\d{8})(\d{2}).nc$")
        with Dataset(filename) as dataset:
            res = []
            for idx in dataset.variables["ensemble_member"][:]:
                res.append(self._get_data_from_dataset(dataset, input_source_types, utc_period,
                                                       geo_location_criteria,
                                                       ensemble_member=idx))
            return res

    @property
    def bounding_box(self):
        # Add a padding to the bounding box to make sure the computational
        # domain is fully enclosed in WRF dataset
        if self._bounding_box is None:
            raise WRFDataRepositoryError("A bounding box must be provided.")
        bounding_box = np.array(self._bounding_box)
        bounding_box[0][0] -= self._x_padding
        bounding_box[0][1] += self._x_padding
        bounding_box[0][2] += self._x_padding
        bounding_box[0][3] -= self._x_padding
        bounding_box[1][0] -= self._y_padding
        bounding_box[1][1] -= self._y_padding
        bounding_box[1][2] += self._y_padding
        bounding_box[1][3] += self._y_padding
        return bounding_box

    def _convert_to_timeseries(self, data):
        """Convert timeseries from numpy structures to shyft.api timeseries.

        We assume the time axis is regular, and that we can use a point time
        series with a parametrized time axis definition and corresponding
        vector of values. If the time series is missing on the data, we insert
        it into non_time_series.

        Returns
        -------
        timeseries: dict
            Time series arrays keyed by type
        """
        tsc = api.TsFactory().create_point_ts
        time_series = {}
        for key, (data, ta) in data.items():
            fslice = (len(data.shape) - 2)*[slice(None)]
            I, J = data.shape[-2:]

            def construct(d):
                if ta.size() != d.size:
                    raise WRFDataRepositoryError("Time axis size {} not equal to the number of "
                                                   "data points ({}) for {}"
                                                   "".format(ta.size(), d.size, key))
                return tsc(ta.size(), ta.start, ta.delta_t,
                           api.DoubleVector_FromNdArray(d.flatten()), api.point_interpretation_policy.POINT_AVERAGE_VALUE)
            time_series[key] = np.array([[construct(data[fslice + [i, j]])
                                          for j in range(J)] for i in range(I)])
        return time_series

    def _limit(self, x, y, data_cs, target_cs):
        """
        Parameters
        ----------
        x: np.ndarray
            X coordinates in meters in cartesian coordinate system
            specified by data_cs
        y: np.ndarray
            Y coordinates in meters in cartesian coordinate system
            specified by data_cs
        data_cs: string
            Proj4 string specifying the cartesian coordinate system
            of x and y
        target_cs: string
            Proj4 string specifying the target coordinate system
        Returns
        -------
        x: np.ndarray
            Coordinates in target coordinate system
        y: np.ndarray
            Coordinates in target coordinate system
        x_mask: np.ndarray
            Boolean index array
        y_mask: np.ndarray
            Boolean index array
        """
        # Get coordinate system for WRF data
        data_proj = Proj(proj=data_cs)
        target_proj = Proj(target_cs)

        # Find bounding box in WRF projection
        bbox = self.bounding_box
        bb_proj = transform(target_proj, data_proj, bbox[0], bbox[1])
        x_min, x_max = min(bb_proj[0]), max(bb_proj[0])
        y_min, y_max = min(bb_proj[1]), max(bb_proj[1])

        # Mask for the limits
        mask = ((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max))

        # Transform from source coordinates to target coordinates
        xx, yy = transform(data_proj, target_proj, x[mask], y[mask])
        return xx, yy, (mask[0], mask[1])

    def _get_data_from_dataset(self, dataset, input_source_types, utc_period,
                               geo_location_criteria, ensemble_member=None):

        if geo_location_criteria is not None:
            self._bounding_box = geo_location_criteria

        if "wind_speed" in input_source_types:
            input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("wind_speed")
            input_source_types.append("x_wind")
            input_source_types.append("y_wind")

        raw_data = {}
        x = dataset.variables.get("XLONG", None)
        y = dataset.variables.get("XLAT", None)
        time = dataset.variables.get("XTIME", None)
        if not all([x, y, time]):
            raise WRFDataRepositoryError("Something is wrong with the dataset."
                                           " x/y coords or time not found.")
        time = convert_netcdf_time(time.units,time)
        # data_cs = dataset.variables.get("projection_lambert", None)
        # TODO: Make sure that "latlong" is the correct coordinate system in WRF data
        #data_cs_proj4 = "+proj=lcc +lon_0=78.9356 +lat_0=31.6857 +lat_1=30 +lat_2=60 +R=6.371e+06 +units=m +no_defs"
        data_cs_proj4 = "latlong"
        if data_cs_proj4 is None:
            raise WRFDataRepositoryError("No coordinate system information in dataset.")

        idx_min = np.searchsorted(time, utc_period.start, side='left')
        idx_max = np.searchsorted(time, utc_period.end, side='right')
        issubset = True if idx_max < len(time) - 1 else False
        time_slice = slice(idx_min, idx_max)
        x, y, (m_x, m_y) = self._limit(x[0], y[0], data_cs_proj4, self.shyft_cs)
        for k in dataset.variables.keys():
            if self.wrf_shyft_map.get(k, None) in input_source_types:
                if k in self._shift_fields and issubset:  # Add one to time slice
                    data_time_slice = slice(time_slice.start, time_slice.stop + 1)
                else:
                    data_time_slice = time_slice
                data = dataset.variables[k]
                dims = data.dimensions
                data_slice = len(data.dimensions)*[slice(None)]
                if ensemble_member is not None:
                    data_slice[dims.index("ensemble_member")] = ensemble_member
                data_slice[dims.index("west_east")] = m_x
                data_slice[dims.index("south_north")] = m_y
                data_slice[dims.index("Time")] = data_time_slice
                pure_arr = data[data_slice]
                if isinstance(pure_arr, np.ma.core.MaskedArray):
                    #print(pure_arr.fill_value)
                    pure_arr = pure_arr.filled(np.nan)
                raw_data[self.wrf_shyft_map[k]] = pure_arr, k
                #raw_data[self.wrf_shyft_map[k]] = np.array(data[data_slice], dtype='d'), k

        if 'HGT' in dataset.variables.keys():
            data = dataset.variables['HGT']
            dims = data.dimensions
            data_slice = len(data.dimensions)*[slice(None)]
            data_slice[dims.index("x")] = m_x
            data_slice[dims.index("y")] = m_y
            z = data[data_slice]
            shp = z.shape
            z = z.reshape(shp[-2], shp[-1])
        else:
            raise WRFDataRepositoryError("No elevations found in dataset.")

        pts = np.dstack((x, y, z)).reshape(*(x.shape + (3,)))

        # Make sure requested fields are valid, and that dataset contains the requested data.
        if not self.allow_subset and not (set(raw_data.keys()).issuperset(input_source_types)):
            raise WRFDataRepositoryError("Could not find all data fields")

        if {"x_wind", "y_wind"}.issubset(raw_data):
            x_wind, _ = raw_data.pop("x_wind")
            y_wind, _ = raw_data.pop("y_wind")
            raw_data["wind_speed"] = np.sqrt(np.square(x_wind) + np.square(y_wind)), "wind_speed"
        extracted_data = self._transform_raw(raw_data, time[time_slice], issubset=issubset)
        return self._geo_ts_to_vec(self._convert_to_timeseries(extracted_data), pts)

    def _transform_raw(self, data, time, issubset=False):
        """
        We need full time if deaccumulating
        """

        def noop_time(t):
            t0 = int(t[0])
            t1 = int(t[1])
            return api.Timeaxis(t0, t1 - t0, len(t))

        def dacc_time(t):
            t0 = int(t[0])
            t1 = int(t[1])
            return noop_time(t) if issubset else api.Timeaxis(t0, t1 - t0, len(t) - 1)

        def noop_space(x):
            return x

        def air_temp_conv(T):
            return T - 273.15

        def prec_conv(p):
            return p[1:]

        def prec_acc_conv(p):
            return np.clip(p[1:] - p[:-1], 0.0, 1000.0)

        def rad_conv(r):
            dr = r[1:] - r[:-1]
            return np.clip(dr/(time[1] - time[0]), 0.0, 5000.0)

        convert_map = {"wind_speed": lambda x, t: (noop_space(x), noop_time(t)),
                       "relative_humidity_2m": lambda x, t: (noop_space(x), noop_time(t)),
                       "air_temperature_2m": lambda x, t: (air_temp_conv(x), noop_time(t)),
                       "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time":
                       lambda x, t: (rad_conv(x), dacc_time(t)),
                       "precipitation_amount": lambda x, t: (prec_conv(x), dacc_time(t)),
                       "precipitation_amount_acc": lambda x, t: (prec_acc_conv(x), dacc_time(t))}
        res = {}
        for k, (v, ak) in data.items():
            res[k] = convert_map[ak](v, time)
        return res

    def _geo_ts_to_vec(self, data, pts):
        res = {}
        for name, ts in iteritems(data):
            tpe = self.source_type_map[name]
            # SiH: Unfortunately, I have not got the boost.python to eat list of non-basic object
            # into the constructor of vectors like this:
            #res[name] = tpe.vector_t([tpe(api.GeoPoint(*pts[idx]), ts[idx]) for idx in np.ndindex(pts.shape[:-1])])
            #     so until then, we have to do the loop
            tpe_v=tpe.vector_t()
            for idx in np.ndindex(pts.shape[:-1]):
                tpe_v.append(tpe(api.GeoPoint(*pts[idx]), ts[idx]))

            res[name] = tpe_v
        return res

    def _get_files(self, t_c, date_pattern):
        utc = api.Calendar()
        file_names = glob(self._filename)
        match_files = []
        match_times = []
        for fn in file_names:
            match = re.search(date_pattern, fn)
            if match:
                datestr, _ , hourstr, _ = match.groups()
                year, month, day = int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8])
                hour = int(hourstr)
                t = utc.time(api.YMDhms(year, month, day, hour))
                if t <= t_c:
                    match_files.append(fn)
                    match_times.append(t)
        if match_files:
            return match_files[np.argsort(match_times)[-1]]
        ymds = utc.calendar_units(t_c)
        date = "{:4d}.{:02d}.{:02d}:{:02d}:{:02d}:{:02d}".format(ymds.year, ymds.month, ymds.day,
                                                                 ymds.hour, ymds.minute, ymds.second)
        raise WRFDataRepositoryError("No matches found for file_pattern = {} and t_c = {} "
                                       "".format(self._filename, date))
