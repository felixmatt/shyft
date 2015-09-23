from __future__ import absolute_import
from __future__ import print_function

import re
from glob import glob
from os import path
from functools import partial
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from pyproj import transform
from shyft import api
from .. import interfaces


class AromeDataRepositoryError(Exception):
    pass


class AromeDataRepository(interfaces.GeoTsRepository):

    def __init__(self, epsg_id, utc_period, directory, filename=None, bounding_box=None,
                 x_padding=5000.0, y_padding=5000.0, elevation_file=None):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP
        model, and initialize data retrieval.

        Parameters
        ----------
        epsg_id: int
            Unique coordinate system id for result coordinates.
            Currently 32632 and 32633 are supperted.
        utc_period: api.UtcPeriod
            Period to fetch such that utc_period.start and utc_period.end are
            both included in the interval, if possible
        directory: string
            Path to directory holding one or possibly more arome data files.
            os.path.isdir(directory) should be true, or exception is raised.
        filename: string, optional
            Name of netcdf file in directory that contains spatially
            distributed input data. Can be a glob pattern as well, in case
            it is used for forecasts or ensambles.
        bounding_box: list, optional
            A list on the form [[x_ul, x_ur, x_lr, x_ll],
            [y_ul, y_ur, y_lr, y_ll]] describing the outer boundaries of the
            domain that shoud be extracted. Coordinates are given in epgs_id
            coordinate system.
        x_padding: float, optional
            Longidutinal padding in meters, added both east and west
        y_padding: float, optional
            Latitudinal padding in meters, added both north and south
        elevation_file: string, optional
            Name of netcdf file of same dimensions in x and y, subject to
            constraints given by bounding box and padding, that contains
            elevation that should be used in stead of elevations in file.


        Arome NWP model output is from:
        http://thredds.met.no/thredds/catalog/arome25/catalog.html

        Contact:
            Name: met.no
            Organization: met.no
            Email: thredds@met.no
            Phone: +47 22 96 30 00
        """
        # Make sure input makes sense, or raise exceptions
        self.directory = directory
        self._filename = None # To be used by forecast and ensemble to read data
        if not path.isdir(self.directory):
            raise interfaces.InterfaceError("No such directory '{}'".format(self.directory))
        self.name_or_pattern = path.join(self.directory, filename)
        if elevation_file is not None:
            self.elevation_file = path.join(self.directory, elevation_file)
            if not path.isfile(self.elevation_file):
                raise interfaces.InterfaceError(
                    "Elevation file '{}' not found".format(self.elevation_file))
        else:
            self.elevation_file = None

        self.epsg_id = epsg_id
        self.shyft_cs = \
            "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(epsg_id - 32600,
                                                                              "WGS84", "WGS84")
        self._x_padding = x_padding
        self._y_padding = y_padding
        self._bounding_box = bounding_box
        # Field names and mappings
        self._netcdf_fields = ["relative_humidity_2m",
                               "air_temperature_2m",
                               "altitude",
                               "precipitation_amount",
                               "x_wind_10m",
                               "y_wind_10m",
                               "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time"]

        self._shyft_fields = ["relative_humidity",
                              "temperature",
                              "z",
                              "precipitation",
                              "x_wind",
                              "y_wind",
                              "radiation"]

        self.source_type_map = {"relative_humidity": api.RelHumSource,
                                "temperature": api.TemperatureSource,
                                "precipitation": api.PrecipitationSource,
                                "radiation": api.RadiationSource,
                                "wind_speed": api.WindSpeedSource}

    @property
    def filename(self):
        if self._filename is not None:
            return self._filename
        elif path.isfile(self.name_or_pattern):
            return self.name_or_pattern
        else:
            match = glob(self.name_or_pattern)
            if len(match) == 1:
                return match[0]
        raise interfaces.InterfaceError("Cannot resolve filename")

    @property
    def bounding_box(self):
        # Add a padding to the bounding box to make sure the computational
        # domain is fully enclosed in arome dataset
        if self._bounding_box is None:
            raise interfaces.InterfaceError("A bounding box must be provided")
        bounding_box = np.array(self._bounding_box)
        bounding_box[0][0] -= self._x_padding
        bounding_box[0][1] += self._x_padding
        bounding_box[0][2] += self._x_padding
        bounding_box[0][3] -= self._x_padding
        bounding_box[1][0] += self._y_padding
        bounding_box[1][1] += self._y_padding
        bounding_box[1][2] -= self._y_padding
        bounding_box[1][3] -= self._y_padding
        return bounding_box

    def _geo_points(self):
        """Return (x,y,z) coordinates for data sources

        Construct and return a numpy array of (x,y,z) coordinates at each
        (i,j) having a data source.
        """
        pts = np.empty(self.xx.shape + (3,), dtype='d')
        pts[:, :, 0] = self.xx
        pts[:, :, 1] = self.yy
        pts[:, :, 2] = self.other_data["z"] if "z" in self.other_data else \
            np.zeros(self.xx.shape, dtype='d')
        return pts

    def _convert_to_timeseries(self, extracted_data):
        """Convert timeseries from numpy structures to shyft.api timeseries.

        We assume the time axis is regular, and that we can use a point time
        series with a parametrized time axis definition and corresponding
        vector of values. If the time series is missing on the data, we insert
        it into non_time_series.

        Returns
        -------
        timeseries: dict
            Time series arrays keyed by type
        non_timeseries: dict
            Other data that can not be converted to time series

        """
        time_series = {}
        non_time_series = {}
        tsc = api.TsFactory().create_point_ts
        for key, (data, ta) in extracted_data.iteritems():
            if ta is None:
                non_time_series[key] = data
                continue
            fslice = (len(data.shape) - 2)*(slice(None),)
            I, J = data.shape[-2:]

            def construct(d):
                return tsc(ta.size(), ta.start(), ta.delta(),
                           api.DoubleVector_FromNdArray(d.flatten()), 0)
            time_series[key] = np.array([[construct(data[fslice + (i, j)])
                                         for j in xrange(J)] for i in xrange(I)])
        return time_series, non_time_series

    def get_timeseries(self, input_source_types, utc_period, geo_location_criteria=None):
        """Get shyft source vectors of time series for input_source_types

        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve (precipitation,temperature..)
        geo_location_criteria: object, optional
            Some type (to be decided), extent (bbox + coord.ref)
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.

        Returns
        -------
        geo_loc_ts: dictionary
            dictionary keyed by ts type, where values are api vectors of geo
            located timeseries.
        """

        if geo_location_criteria is not None:
            self._bounding_box = geo_location_criteria

        if not isinstance(utc_period, api.UtcPeriod):
            utc_period = api.UtcPeriod(utc_period[0]. utc_period[1])

        if "wind_speed" in input_source_types:
            input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("wind_speed")
            input_source_types.append("x_wind")
            input_source_types.append("y_wind")

        # Open netcdf dataset. TODO: use with...
        if not path.isfile(self.filename):
            raise interfaces.InterfaceError("File '{}' not found".format(self.filename))
        dataset = Dataset(self.filename)
        data_vars = dataset.variables

        # Extract time dimension
        time = data_vars["time"][:]
        dts = time[1:] - time[:-1]

        idx_min = time.searchsorted(utc_period.start, side='left')
        idx_max = time.searchsorted(utc_period.end, side='right')
        time_slice = slice(idx_min, idx_max)
        extract_subset = True if time[time_slice].shape != time.shape else False
        time = time[time_slice]

        # arome data and time conversions, ordered as _netcdf_fields
        def netcdf_data_convert(t):
            """
            For a given utc time list t, return a list of callable tuples to
            convert from arome data to shyft data. For radiation we calculate:
            rad[t_i] = sw_flux(t_{i+1}) - sw_flux(t_i)/dt for i in 0, ..., N-1,
            where N is the number of values in the dataset, and equals the
            number of forcast time points + 1. Also temperatures are converted
            from Kelvin to Celcius, and the elevation data set is treated as a
            special case.
            """

            def t_to_ta(t, shift):
                if extract_subset:
                    shift = 0
                return api.Timeaxis(int(t[0]), int(t[1] - t[0]), len(t) - shift)

            def noop(d):
                return d[time_slice]

            def prec(d):
                return d[1:][time_slice]
            t_to_ta_0 = partial(t_to_ta, t, 0)  # Full
            t_to_ta_1 = partial(t_to_ta, t, 1)
            return [(noop, t_to_ta_0),
                    (lambda air_temp: air_temp[time_slice] - 273.15, t_to_ta_0),
                    (lambda x: x, lambda: None),  # Altitude
                    (prec, t_to_ta_1),
                    (noop, t_to_ta_0),
                    (noop, t_to_ta_0),
                    (lambda rad: np.clip(((rad[1:][time_slice] -
                                           rad[:-1][time_slice])/(dts[time_slice,
                                                                      np.newaxis,
                                                                      np.newaxis,
                                                                      np.newaxis])), 0.0, 1000.0),
                     t_to_ta_1)]

        if input_source_types is None:
            input_source_types = self._shyft_fields
        else:
            assert set(input_source_types).issubset(self._shyft_fields)  # TODO: Check

        shyft_net_map = {s: n for n, s in zip(self._netcdf_fields, self._shyft_fields)}
        data_convert_map = {s: c for s, c in zip(self._shyft_fields, netcdf_data_convert(time))}

        # Target projection
        shyft_proj = Proj(self.shyft_cs)

        def find_inds(data_vars, data):
            # Get coordinate system for arome data
            data_cs = str(data_vars[data.grid_mapping].proj4)
            data_cs += " +towgs84=0,0,0"
            data_proj = Proj(data_cs)

            # Find bounding box in arome projection
            bbox = self.bounding_box
            bb_proj = transform(shyft_proj, data_proj,
                                bbox[0], bbox[1])
            x_min, x_max = min(bb_proj[0]), max(bb_proj[0])
            y_min, y_max = min(bb_proj[1]), max(bb_proj[1])

            # Limit data
            x = data_vars["x"][:]
            x1 = np.where(x >= x_min)[0]
            x2 = np.where(x <= x_max)[0]
            x_inds = np.intersect1d(x1, x2, assume_unique=True)

            y = data_vars["y"][:]
            y1 = np.where(y >= y_min)[0]
            y2 = np.where(y <= y_max)[0]
            y_inds = np.intersect1d(y1, y2, assume_unique=True)

            # Transform from arome coordinates to shyft coordinates
            self._ox, self._oy = np.meshgrid(x[x_inds], y[y_inds])
            xx, yy = transform(data_proj, shyft_proj, *np.meshgrid(x[x_inds], y[y_inds]))

            return xx, yy, x_inds, y_inds

        raw_data = {}
        inds_found = False
        self.xx = self.yy = self.extracted_data = None
        for data_field in input_source_types:
            if not shyft_net_map[data_field] in data_vars.keys():
                continue
            data = data_vars[shyft_net_map[data_field]]

            if not inds_found:
                self.xx, self.yy, x_inds, y_inds = find_inds(data_vars, data)
                inds_found = True

            # Construct slice
            data_slice = len(data.dimensions)*[slice(None)]
            data_slice[data.dimensions.index("x")] = x_inds
            data_slice[data.dimensions.index("y")] = y_inds

            # Add extracted data and corresponding coordinates to class
            raw_data[data_field] = data[data_slice]

        extracted_data = {key: (data_convert_map[key][0](raw_data[key]),
                                data_convert_map[key][1]()) for key in raw_data}

        if "x_wind" in extracted_data.keys() and "y_wind" in extracted_data.keys():
            x_wind, _ = extracted_data.pop("x_wind")
            y_wind, t = extracted_data.pop("y_wind")
            extracted_data["wind_speed"] = np.sqrt(np.square(x_wind) + np.square(y_wind)), t

        if self.elevation_file is not None:
            ds2 = Dataset(self.elevation_file)
            data = ds2.variables["altitude"]
            if "altitude" not in ds2.variables.keys():
                raise interfaces.InterfaceError(
                    "File '{}' does not contain altitudes".format(self.elevation_file))
            xx, yy, x_inds, y_inds = find_inds(ds2.variables, data)
            data_slice = len(data.dimensions)*[slice(None)]
            data_slice[data.dimensions.index("x")] = x_inds
            data_slice[data.dimensions.index("y")] = y_inds
            assert np.linalg.norm(self.xx - xx) < 1.0e-10
            assert np.linalg.norm(self.yy - yy) < 1.0e-10
            extracted_data["z"] = data[data_slice], None

        self.time_series, self.other_data = self._convert_to_timeseries(extracted_data)

        if "geo_points" not in self.other_data:
            self.other_data["geo_points"] = self._geo_points()
        pts = self.other_data["geo_points"]
        sources = {}
        all_ = slice(None)
        for key, ts in self.time_series.iteritems():
            if key not in self.source_type_map:
                continue
            tpe = self.source_type_map[key]
            sources[key] = tpe.vector_t([tpe(api.GeoPoint(*pts[idx + (all_,)]),
                                         ts[idx]) for idx in
                                         np.ndindex(pts.shape[:-1])])
        return sources

    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria=None):
        """
        See base class
        """
        utc = api.Calendar()
        file_names = glob(self.name_or_pattern)
        match_files = []
        match_times = []
        for fn in file_names:
            match = re.search("_(\d{8})_(\d{2}).nc$", fn)
            if match:
                datestr, hourstr = match.groups()
                year, month, day = int(datestr[:4]), int(datestr[4:6]), int(datestr[6:8])
                hour = int(hourstr)
                t = utc.time(api.YMDhms(year, month, day, hour))
                if t <= t_c:
                    match_files.append(fn)
                    match_times.append(t)
        if match_files:
            self._filename = match_files[np.argsort(match_times)[-1]]
        res = self.get_timeseries(input_source_types, utc_period, geo_location_criteria)
        self._filename = None
        return res

    def get_forecast_ensemble(self, input_source_types, utc_period, t_c,
                              geo_location_criteria=None):
        """
        """
        raise interfaces.InterfaceError()
