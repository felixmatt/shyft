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

    _G = 9.80665  # WMO-defined gravity constant to calculate the height in metres from geopotential

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
            "T2": "temperature",
            "HGT": "z",
            "PREC_ACC_NC": "precipitation",
            "U10": "x_wind",
            "V10": "y_wind",
            "SWDOWN": "radiation",
            "Q2": "mixing_ratio",
            "PSFC": "pressure"}

        # Fields that need an additional timeslice because the measure average values
        # self._shift_fields = ("PREC_ACC_NC", "SWDOWN")
        self._shift_fields = ()

        self.source_type_map = {"relative_humidity": api.RelHumSource,
                                "temperature": api.TemperatureSource,
                                "precipitation": api.PrecipitationSource,
                                "radiation": api.RadiationSource,
                                "wind_speed": api.WindSpeedSource}

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
                # TODO: We still need to adapt this to the pattern structure of multiple WRF files
                filename = self._get_files(utc_period.start, "_(\d{8})([T_])(\d{2})(Z)?.nc$")
            else:
                raise WRFDataRepositoryError("File '{}' not found".format(filename))
        with Dataset(filename) as dataset:
            return self._get_data_from_dataset(dataset, input_source_types,
                                               utc_period, geo_location_criteria)

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
            def construct(d):
                if ta.size() != d.size:
                    raise WRFDataRepositoryError("Time axis size {} not equal to the number of "
                                                 "data points ({}) for {}"
                                                 "".format(ta.size(), d.size, key))
                return tsc(ta.size(), ta.start, ta.delta_t,
                           api.DoubleVector_FromNdArray(d), api.point_interpretation_policy.POINT_AVERAGE_VALUE)

            time_series[key] = np.array([construct(data[:, i]) for i in range(data.shape[1])])
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
        x_targ, y_targ = transform(data_proj, target_proj, x, y)
        bbox = self.bounding_box
        x_min, x_max = min(bbox[0]), max(bbox[0])
        y_min, y_max = min(bbox[1]), max(bbox[1])
        # Mask for the limits
        mask = ((x_targ >= x_min) & (x_targ <= x_max) & (y_targ >= y_min) & (y_targ <= y_max))

        if not mask.any():
            raise WRFDataRepositoryError("Bounding box doesn't intersect with dataset.")

        # Transform from source coordinates to target coordinates
        xx, yy = x_targ[mask], y_targ[mask]
        return xx, yy, mask

    def _calculate_rel_hum(self, T2, PSFC, Q2):
        # constants
        EZERO = 6.112
        ESLCON1 = 17.67
        ESLCON2 = 29.65
        CELKEL = 273.15
        RD = 287.
        RV = 461.6
        EPS = 0.622

        # calculation
        RH = np.empty_like(T2)
        es = EZERO * np.exp(ESLCON1 * (T2 - CELKEL) / (T2 - ESLCON2))
        qvs = EPS * es / (0.01 * PSFC - (1.0 - EPS) * es)
        RH = Q2 / qvs
        RH[RH > 1.0] = 1.0
        RH[RH < 0.0] = 0.0
        return RH

    def _get_data_from_dataset(self, dataset, input_source_types, utc_period,
                               geo_location_criteria, ensemble_member=None):

        if geo_location_criteria is not None:
            self._bounding_box = geo_location_criteria

        input_source_types_orig = list(input_source_types)
        if "wind_speed" in input_source_types:
            input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("wind_speed")
            input_source_types.append("x_wind")
            input_source_types.append("y_wind")

        if "relative_humidity" in input_source_types:
            input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("relative_humidity")
            input_source_types.append("mixing_ratio")
            input_source_types.append("pressure")
            if not "temperature" in input_source_types:
                input_source_types.append("temperature")  # Needed for rel_hum calculation

        raw_data = {}
        x = dataset.variables.get("XLONG", None)
        y = dataset.variables.get("XLAT", None)
        time = dataset.variables.get("XTIME", None)
        if not all([x, y, time]):
            raise WRFDataRepositoryError("Something is wrong with the dataset."
                                         " x/y coords or time not found.")
        # TODO: make a check that dim1 is time, dim2 is ..., dim3 is ...
        x = x[0, :, :].reshape(x.shape[1] * x.shape[2])
        y = y[0, :, :].reshape(y.shape[1] * y.shape[2])
        time = convert_netcdf_time(time.units, time)
        # TODO: Make sure that "latlong" is the correct coordinate system in WRF data
        # data_cs_proj4 = "+proj=lcc +lon_0=78.9356 +lat_0=31.6857 +lat_1=30 +lat_2=60 +R=6.371e+06 +units=m +no_defs"
        data_cs_proj4 = "latlong"
        if data_cs_proj4 is None:
            raise WRFDataRepositoryError("No coordinate system information in dataset.")

        idx_min = np.searchsorted(time, utc_period.start, side='left')
        idx_max = np.searchsorted(time, utc_period.end, side='right')
        issubset = True if idx_max < len(time) - 1 else False
        time_slice = slice(idx_min, idx_max)
        x, y, mask = self._limit(x, y, data_cs_proj4, self.shyft_cs)
        for k in dataset.variables.keys():
            if self.wrf_shyft_map.get(k, None) in input_source_types:
                if k in self._shift_fields and issubset:  # Add one to time slice
                    data_time_slice = slice(time_slice.start, time_slice.stop + 1)
                else:
                    data_time_slice = time_slice
                data = dataset.variables[k]
                # TODO: make a check that data's dim1 is time, dim2 is ..., dim3 is ...
                data = data[:].reshape(data.shape[0], data.shape[1] * data.shape[2])
                data_slice = [slice(None), slice(None)]
                # if ensemble_member is not None:
                #    data_slice[dims.index("ensemble_member")] = ensemble_member
                data_slice[0] = data_time_slice
                data_slice[1] = mask
                pure_arr = data[data_slice]
                if isinstance(pure_arr, np.ma.core.MaskedArray):
                    pure_arr = pure_arr.filled(np.nan)
                raw_data[self.wrf_shyft_map[k]] = pure_arr, k

        if 'HGT' in dataset.variables.keys():
            data = dataset.variables['HGT']
            data = data[0, :, :].reshape(data.shape[1] * data.shape[2])  # get the first entry in time
            z = data[mask]
        else:
            raise WRFDataRepositoryError("No elevations found in dataset.")

        pts = np.stack((x, y, z), axis=-1)

        # Make sure requested fields are valid, and that dataset contains the requested data.
        if not self.allow_subset and not (set(raw_data.keys()).issuperset(input_source_types)):
            raise WRFDataRepositoryError("Could not find all data fields")
        if {"x_wind", "y_wind"}.issubset(raw_data):
            x_wind, _ = raw_data.pop("x_wind")
            y_wind, _ = raw_data.pop("y_wind")
            raw_data["wind_speed"] = np.sqrt(np.square(x_wind) + np.square(y_wind)), "wind_speed"
        if {"temperature", "pressure", "mixing_ratio"}.issubset(raw_data):
            pressure, _ = raw_data.pop("pressure")
            mixing_ratio, _ = raw_data.pop("mixing_ratio")
            if "temperature" in input_source_types_orig:
                temperature, _ = raw_data["temperature"]  # Temperature input requested
            else:
                temperature, _ = raw_data.pop("temperature")  # Temperature only needed for relhum calculation
            raw_data["relative_humidity"] = self._calculate_rel_hum(temperature, pressure,
                                                                    mixing_ratio), "relative_humidity_2m"
        extracted_data = self._transform_raw(raw_data, time[time_slice], issubset=issubset)
        return self._geo_ts_to_vec(self._convert_to_timeseries(extracted_data), pts)

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
            return T - 273.16  # definition says -273.15, but regression test says -273.16..

        def prec_conv(p):
            # return p[1:]
            return p

        # def prec_acc_conv(p):
        #    return np.clip(p[1:] - p[:-1], 0.0, 1000.0)

        def rad_conv(r):
            # dr = r[1:] - r[:-1]
            # return np.clip(dr/(time[1] - time[0]), 0.0, 5000.0)
            return r

        convert_map = {"wind_speed": lambda x, t: (noop_space(x), noop_time(t)),
                       "relative_humidity_2m": lambda x, t: (noop_space(x), noop_time(t)),
                       "T2": lambda x, t: (air_temp_conv(x), noop_time(t)),
                       "SWDOWN": lambda x, t: (rad_conv(x), noop_time(t)),
                       "PREC_ACC_NC": lambda x, t: (prec_conv(x), noop_time(t))}
        # "precipitation_amount_acc": lambda x, t: (prec_acc_conv(x), dacc_time(t))}
        res = {}
        for k, (v, ak) in data.items():
            res[k] = convert_map[ak](v, time)
        return res

    def _geo_ts_to_vec(self, data, pts):
        res = {}
        for name, ts in data.items():
            tpe = self.source_type_map[name]
            # SiH: Unfortunately, I have not got the boost.python to eat list of non-basic object
            # into the constructor of vectors like this:
            # res[name] = tpe.vector_t([tpe(api.GeoPoint(*pts[idx]), ts[idx]) for idx in np.ndindex(pts.shape[:-1])])
            #     so until then, we have to do the loop
            tpe_v = tpe.vector_t()
            for idx in range(len(ts)):
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
                datestr, _, hourstr, _ = match.groups()
                year, month, day = int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8])
                hour = int(hourstr)
                t = utc.time(year, month, day, hour)
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
