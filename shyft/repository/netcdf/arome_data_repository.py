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

class AromeDataRepositoryError(Exception):
    pass


class AromeDataRepository(interfaces.GeoTsRepository):
    """
    Repository for geo located timeseries given as Arome(*) data in
    netCDF files.

    NetCDF dataset assumptions:
        * Root group has variables:
            * time: timestamp (int) array with seconds since epoc
                    (1970.01.01 00:00, UTC) for each data point
            * x: float array of latitudes
            * y: float array of longitudes
        * Root group has subset of variables:
            * relative_humidity_2m: float array of dims (time, 1, y, x)
            * air_temperature_2m: float array of dims (time, 1, y, x)
            * altitude: float array of dims (y, x)
            * precipitation_amount: float array of dims (time, y, x)
            * x_wind_10m: float array of dims (time, y, x)
            * y_wind_10m: float array of dims (time, y, x)
            * integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time:
              float array of dims (time, 1, y, x)
            * All variables are assumed to have the attribute grid_mapping
              which should be a reference to a variable in the root group
              that has an attribute named proj4. Example code:
                ds = netCDF4.Dataset(arome_file)
                var = "precipitation_amount"
                mapping = ds.variables[var].grid_mapping
                proj = ds.variables[mapping].proj4


    (*) Arome NWP model output is from:
        http://thredds.met.no/thredds/catalog/arome25/catalog.html

        Contact:
            Name: met.no
            Organization: met.no
            Email: thredds@met.no
            Phone: +47 22 96 30 00

    """

    _G = 9.80665 #  WMO-defined gravity constant to calculate the height in metres from geopotential

    def __init__(self, epsg, directory, filename=None, bounding_box=None,
                 x_padding=5000.0, y_padding=5000.0, elevation_file=None, allow_subset=False):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP model,
        and initialize data retrieval.

        Parameters
        ----------
        epsg: string
            Unique coordinate system id for result coordinates.
            Currently "32632" and "32633" are supperted.
        directory: string
            Path to directory holding one or possibly more arome data files.
            os.path.isdir(directory) should be true, or exception is raised.
        filename: string, optional
            Name of netcdf file in directory that contains spatially
            distributed input data. Can be a glob pattern as well, in case
            it is used for forecasts or ensambles.
        bounding_box: list, optional
            A list on the form:
            [[x_ll, x_lr, x_ur, x_ul],
             [y_ll, y_lr, y_ur, y_ul]],
            describing the outer boundaries of the domain that shoud be
            extracted. Coordinates are given in epsg coordinate system.
        x_padding: float, optional
            Longidutinal padding in meters, added both east and west
        y_padding: float, optional
            Latitudinal padding in meters, added both north and south
        elevation_file: string, optional
            Name of netcdf file of same dimensions in x and y, subject to
            constraints given by bounding box and padding, that contains
            elevation that should be used in stead of elevations in file.
        allow_subset: bool
            Allow extraction of a subset of the given source fields
            instead of raising exception.
        """
        #directory = directory.replace('${SHYFTDATA}', os.getenv('SHYFTDATA', '.'))
        directory = path.expandvars(directory)
        self._filename = path.join(directory, filename)
        self.allow_subset = allow_subset
        if not path.isdir(directory):
            raise AromeDataRepositoryError("No such directory '{}'".format(directory))

        if elevation_file is not None:
            self.elevation_file = path.join(directory, elevation_file)
            if not path.isfile(self.elevation_file):
                raise AromeDataRepositoryError(
                    "Elevation file '{}' not found".format(self.elevation_file))
        else:
            self.elevation_file = None

        self.shyft_cs = "+init=EPSG:{}".format(epsg)
        self._x_padding = x_padding
        self._y_padding = y_padding
        self._bounding_box = bounding_box

        # Field names and mappings
        self._arome_shyft_map = {"relative_humidity_2m": "relative_humidity",
                                 "air_temperature_2m": "temperature",
                                 "altitude": "z",
                                 "precipitation_amount": "precipitation",
                                 "precipitation_amount_acc": "precipitation",
                                 "x_wind_10m": "x_wind",
                                 "y_wind_10m": "y_wind",
                                 "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time":
                                 "radiation"}

        self.var_units = {"air_temperature_2m": ['K'],
                          "relative_humidity_2m": ['1'],
                          "precipitation_amount_acc": ['kg/m^2'],
                          "precipitation_amount": ['kg/m^2'],
                          "x_wind_10m": ['m/s'],
                          "y_wind_10m": ['m/s'],
                          "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time": ['W s/m^2']}

        self._shift_fields = ("precipitation_amount", "precipitation_amount_acc",
                              "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time")

        self.source_type_map = {"relative_humidity": api.RelHumSource,
                                "temperature": api.TemperatureSource,
                                "precipitation": api.PrecipitationSource,
                                "radiation": api.RadiationSource,
                                "wind_speed": api.WindSpeedSource}

        self.series_type = {"relative_humidity": api.POINT_INSTANT_VALUE,
                            "temperature": api.POINT_INSTANT_VALUE,
                            "precipitation": api.POINT_AVERAGE_VALUE,
                            "radiation": api.POINT_AVERAGE_VALUE,
                            "wind_speed": api.POINT_INSTANT_VALUE}

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
                raise AromeDataRepositoryError("File '{}' not found".format(filename))
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
        # domain is fully enclosed in arome dataset
        if self._bounding_box is None:
            raise AromeDataRepositoryError("A bounding box must be provided.")
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
                    raise AromeDataRepositoryError("Time axis size {} not equal to the number of "
                                                   "data points ({}) for {}"
                                                   "".format(ta.size(), d.size, key))
                return tsc(ta.size(), ta.start, ta.delta_t,
                           api.DoubleVector_FromNdArray(d.flatten()), self.series_type[key])
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
        # Get coordinate system for arome data
        data_proj = Proj(data_cs)
        target_proj = Proj(target_cs)

        # Find bounding box in arome projection
        bbox = self.bounding_box
        bb_proj = transform(target_proj, data_proj, bbox[0], bbox[1])
        x_min, x_max = min(bb_proj[0]), max(bb_proj[0])
        y_min, y_max = min(bb_proj[1]), max(bb_proj[1])

        # Limit data
        x_upper = x >= x_min
        x_lower = x <= x_max
        y_upper = y >= y_min
        y_lower = y <= y_max
        if sum(x_upper == x_lower) < 2:
            if sum(x_lower) == 0 and sum(x_upper) == len(x_upper):
                raise AromeDataRepositoryError("Bounding box longitudes don't intersect with dataset.")
            x_upper[np.argmax(x_upper) - 1] = True
            x_lower[np.argmin(x_lower)] = True
        if sum(y_upper == y_lower) < 2:
            if sum(y_lower) == 0 and sum(y_upper) == len(y_upper):
                raise AromeDataRepositoryError("Bounding box latitudes don't intersect with dataset.")
            y_upper[np.argmax(y_upper) - 1] = True
            y_lower[np.argmin(y_lower)] = True

        x_inds = np.nonzero(x_upper == x_lower)[0]
        y_inds = np.nonzero(y_upper == y_lower)[0]

        # Masks
        x_mask = x_upper == x_lower
        y_mask = y_upper == y_lower

        # Transform from source coordinates to target coordinates
        xx, yy = transform(data_proj, target_proj, *np.meshgrid(x[x_mask], y[y_mask]))

        return xx, yy, (x_mask, y_mask), (x_inds, y_inds)

    def _get_data_from_dataset(self, dataset, input_source_types, utc_period,
                               geo_location_criteria, ensemble_member=None):

        if geo_location_criteria is not None:
            self._bounding_box = geo_location_criteria

        if "wind_speed" in input_source_types:
            input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("wind_speed")
            input_source_types.append("x_wind")
            input_source_types.append("y_wind")

        unit_ok = {k: dataset.variables[k].units in self.var_units[k]
                   for k in dataset.variables.keys() if self._arome_shyft_map.get(k, None) in input_source_types}
        if not all(unit_ok.values()):
            raise AromeDataRepositoryError("The following variables have wrong unit: {}.".format(
                ', '.join([k for k, v in unit_ok.items() if not v])))

        raw_data = {}
        x = dataset.variables.get("x", None)
        y = dataset.variables.get("y", None)
        time = dataset.variables.get("time", None)
        if not all([x, y, time]):
            raise AromeDataRepositoryError("Something is wrong with the dataset."
                                           " x/y coords or time not found.")
        if not all([var.units in ['km', 'm'] for var in [x, y]]) and x.units == y.units:
            raise AromeDataRepositoryError("The unit for x and y coordinates should be either m or km.")
        coord_conv = 1.
        if x.units == 'km':
            coord_conv = 1000.
        time = convert_netcdf_time(time.units,time)
        data_cs = dataset.variables.get("projection_lambert", None)
        if data_cs is None:
            raise AromeDataRepositoryError("No coordinate system information in dataset.")

        idx_min = np.searchsorted(time, utc_period.start, side='left')
        idx_max = np.searchsorted(time, utc_period.end, side='right')
        issubset = True if idx_max < len(time) - 1 else False
        time_slice = slice(idx_min, idx_max)
        x, y, (m_x, m_y), _ = self._limit(x[:]*coord_conv, y[:]*coord_conv, data_cs.proj4, self.shyft_cs)
        for k in dataset.variables.keys():
            if self._arome_shyft_map.get(k, None) in input_source_types:
                if k in self._shift_fields and issubset:  # Add one to time slice
                    data_time_slice = slice(time_slice.start, time_slice.stop + 1)
                else:
                    data_time_slice = time_slice
                data = dataset.variables[k]
                dims = data.dimensions
                data_slice = len(data.dimensions)*[slice(None)]
                if ensemble_member is not None:
                    data_slice[dims.index("ensemble_member")] = ensemble_member
                data_slice[dims.index("x")] = m_x
                data_slice[dims.index("y")] = m_y
                data_slice[dims.index("time")] = data_time_slice
                pure_arr = data[data_slice]
                if isinstance(pure_arr, np.ma.core.MaskedArray):
                    #print(pure_arr.fill_value)
                    pure_arr = pure_arr.filled(np.nan)
                raw_data[self._arome_shyft_map[k]] = pure_arr, k
                #raw_data[self._arome_shyft_map[k]] = np.array(data[data_slice], dtype='d'), k

        if self.elevation_file is not None:
            _x, _y, z = self._read_elevation_file(self.elevation_file)
            assert np.linalg.norm(x - _x) < 1.0e-10  # x/y coordinates should match
            assert np.linalg.norm(y - _y) < 1.0e-10
        elif any([nm in dataset.variables.keys() for nm in ['altitude', 'surface_geopotential']]):
            var_nm = ['altitude', 'surface_geopotential'][[nm in dataset.variables.keys() for nm in ['altitude', 'surface_geopotential']].index(True)]
            data = dataset.variables[var_nm]
            dims = data.dimensions
            data_slice = len(data.dimensions)*[slice(None)]
            data_slice[dims.index("x")] = m_x
            data_slice[dims.index("y")] = m_y
            z = data[data_slice]
            shp = z.shape
            z = z.reshape(shp[-2], shp[-1])
            if var_nm == 'surface_geopotential':
                z /= self._G
        else:
            raise AromeDataRepositoryError("No elevations found in dataset"
                                           ", and no elevation file given.")

        pts = np.dstack((x, y, z)).reshape(*(x.shape + (3,)))

        # Make sure requested fields are valid, and that dataset contains the requested data.
        if not self.allow_subset and not (set(raw_data.keys()).issuperset(input_source_types)):
            raise AromeDataRepositoryError("Could not find all data fields")

        if set(("x_wind", "y_wind")).issubset(raw_data):
            x_wind, _ = raw_data.pop("x_wind")
            y_wind, _ = raw_data.pop("y_wind")
            raw_data["wind_speed"] = np.sqrt(np.square(x_wind) + np.square(y_wind)), "wind_speed"
        extracted_data = self._transform_raw(raw_data, time[time_slice], issubset=issubset)
        return self._geo_ts_to_vec(self._convert_to_timeseries(extracted_data), pts)

    def _read_elevation_file(self, filename):
        with Dataset(self.elevation_file) as dataset:
            elev = dataset.variables["altitude"]
            if "altitude" not in dataset.variables.keys():
                raise interfaces.InterfaceError(
                    "File '{}' does not contain altitudes".format(self.elevation_file))
            x, y, (x_mask, y_mask), _ = \
                self._limit(dataset.variables.pop("x"),
                            dataset.variables.pop("y"),
                            dataset.variables.pop(elev.grid_mapping).proj4,
                            self.shyft_cs)
            data_slice = len(elev.dimensions)*[slice(None)]
            data_slice[elev.dimensions.index("x")] = x_mask
            data_slice[elev.dimensions.index("y")] = y_mask
            return x, y, elev[data_slice]

    def _transform_raw(self, data, time, issubset=False):
        """
        We need full time if deaccumulating
        """

        def noop_time(t):
            t0 = int(t[0])
            t1 = int(t[1])
            return api.TimeAxisFixedDeltaT(t0, t1 - t0, len(t))

        def dacc_time(t):
            t0 = int(t[0])
            t1 = int(t[1])
            return noop_time(t) if issubset else api.TimeAxisFixedDeltaT(t0, t1 - t0, len(t) - 1)

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
        raise AromeDataRepositoryError("No matches found for file_pattern = {} and t_c = {} "
                                       "".format(self._filename, date))
