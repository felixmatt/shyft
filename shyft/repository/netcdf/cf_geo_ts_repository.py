from __future__ import absolute_import
from __future__ import print_function
from six import iteritems
from builtins import range


from os import path
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from pyproj import transform
from shyft import api
from .. import interfaces
from .time_conversion import convert_netcdf_time

class CFDataRepositoryError(Exception):
    pass


class CFDataRepository(interfaces.GeoTsRepository):
    """
    Repository for geo located timeseries stored in netCDF files.

    """
                     
    def __init__(self, params, region_config):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP model,
        and initialize data retrieval.
        """
        self._rconf = region_config
        epsg = self._rconf.domain()["EPSG"]
        directory = params['data_dir']
        filename = params["stations_met"]
        
        if not path.isdir(directory):
            raise CFDataRepositoryError("No such directory '{}'".format(directory))
        if not path.isabs(filename):
            # Relative paths will be prepended the data_dir
            filename = path.join(directory, filename)
        if not path.isfile(filename):
            raise CFDataRepositoryError("No such file '{}'".format(filename))
            
        self._filename = filename # path.join(directory, filename)
        self.allow_subset = True # allow_subset

        
        self.elevation_file = None

        self.shyft_cs = "+init=EPSG:{}".format(epsg)
        self._x_padding = 5000.0 # x_padding
        self._y_padding = 5000.0 # y_padding
        self._bounding_box = None # bounding_box

        # Field names and mappings netcdf_name: shyft_name
        self._nc_shyft_map = {"relative_humidity": "relative_humidity",
                                 "temperature": "temperature",
                                 "z": "z",
                                 "precipitation": "precipitation",
                                 "precipitation_amount_acc": "precipitation",
                                 "wind_speed": "wind_speed",
                                 "global_radiation":"radiation"}

        self._shift_fields = ("precipitation_amount_acc",
                              "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time")

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
            raise CFDataRepositoryError("File '{}' not found".format(filename))
        with Dataset(filename) as dataset:
            return self._get_data_from_dataset(dataset, input_source_types,
                                               utc_period, geo_location_criteria)

    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria):
        """
        Parameters:
        see get_timeseries
        semantics for utc_period: Get the forecast closest up to utc_period.start
        """
        raise NotImplementedError("get_forecast")

    def get_forecast_ensemble(self, input_source_types, utc_period,
                              t_c, geo_location_criteria=None):
        raise NotImplementedError("get_forecast_ensemble")

    @property
    def bounding_box(self):
        # Add a padding to the bounding box to make sure the computational
        # domain is fully enclosed in arome dataset
        if self._bounding_box is None:
            raise CFDataRepositoryError("A bounding box must be provided.")
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
                    raise CFDataRepositoryError("Time axis size {} not equal to the number of "
                                                   "data points ({}) for {}"
                                                   "".format(ta.size(), d.size, key))
                return tsc(ta.size(), ta.start(), ta.delta(),
                           api.DoubleVector_FromNdArray(d.flatten()), 0)
            #time_series[key] = np.array([[construct(data[fslice + [i, j]])
            #                              for j in range(J)] for i in range(I)])
            time_series[key] = np.array([construct(data[:,j]) for j in range(J)])                                
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
        # Get coordinate system for netcdf data
        data_proj = Proj(data_cs)
        target_proj = Proj(target_cs)

        # Find bounding box in netcdf projection
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
                raise CFDataRepositoryError("Bounding box longitudes don't intersect with dataset.")
            x_upper[np.argmax(x_upper) - 1] = True
            x_lower[np.argmin(x_lower)] = True
        if sum(y_upper == y_lower) < 2:
            if sum(y_lower) == 0 and sum(y_upper) == len(y_upper):
                raise CFDataRepositoryError("Bounding box latitudes don't intersect with dataset.")
            y_upper[np.argmax(y_upper) - 1] = True
            y_lower[np.argmin(y_lower)] = True

        x_inds = np.nonzero(x_upper == x_lower)[0]
        y_inds = np.nonzero(y_upper == y_lower)[0]

        # Masks
        x_mask = x_upper == x_lower
        y_mask = y_upper == y_lower
        xy_mask = ((x_mask)&(y_mask))

        # Transform from source coordinates to target coordinates
        xx, yy = transform(data_proj, target_proj, x[xy_mask], y[xy_mask])

        return xx, yy, xy_mask, (x_inds, y_inds)

    def _get_data_from_dataset(self, dataset, input_source_types, utc_period,
                               geo_location_criteria, ensemble_member=None):

        if geo_location_criteria is not None:
            self._bounding_box = geo_location_criteria

        raw_data = {}
        x = dataset.variables.get("x", None)
        y = dataset.variables.get("y", None)
        time = dataset.variables.get("time", None)
        if not all([x, y, time]):
            raise CFDataRepositoryError("Something is wrong with the dataset."
                                           " x/y coords or time not found.")
        time = convert_netcdf_time(time.units,time)
        data_cs = dataset.variables.get("crs", None)
        if data_cs is None:
            raise CFDataRepositoryError("No coordinate system information in dataset.")

        idx_min = np.searchsorted(time, utc_period.start, side='left')
        idx_max = np.searchsorted(time, utc_period.end, side='right')

        issubset = True if idx_max < len(time) - 1 else False
        time_slice = slice(idx_min, idx_max)

        x, y, m_xy, _ = self._limit(x[:], y[:], data_cs.proj4, self.shyft_cs)
        for k in dataset.variables.keys():
            if self._nc_shyft_map.get(k, None) in input_source_types:
                if k in self._shift_fields and issubset:  # Add one to time slice
                    data_time_slice = slice(time_slice.start, time_slice.stop + 1)
                else:
                    data_time_slice = time_slice
                data = dataset.variables[k]
                dims = data.dimensions
                data_slice = len(data.dimensions)*[slice(None)]
                if ensemble_member is not None:
                    data_slice[dims.index("ensemble_member")] = ensemble_member
                data_slice[dims.index("station")] = m_xy
                data_slice[dims.index("time")] = data_time_slice
                pure_arr = data[data_slice]
                if isinstance(pure_arr, np.ma.core.MaskedArray):
                    #print(pure_arr.fill_value)
                    pure_arr = pure_arr.filled(np.nan)
                raw_data[self._nc_shyft_map[k]] = pure_arr, k
                #raw_data[self._nc_shyft_map[k]] = np.array(data[data_slice], dtype='d'), k

        if "z" in dataset.variables.keys():
            data = dataset.variables["z"]
            #dims = data.dimensions
            #data_slice = len(data.dimensions)*[slice(None)]
            #data_slice[dims.index("station")] = m_xy
            #z = data[data_slice]
            z = data[m_xy]
        else:
            raise CFDataRepositoryError("No elevations found in dataset")

        pts = np.dstack((x, y, z)).reshape(-1,3)
        self.pts = pts

        # Make sure requested fields are valid, and that dataset contains the requested data.
        if not self.allow_subset and not (set(raw_data.keys()).issuperset(input_source_types)):
            raise CFDataRepositoryError("Could not find all data fields")

        extracted_data = self._transform_raw(raw_data, time[time_slice], issubset=issubset)
        self.extracted_data = extracted_data
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
                       "relative_humidity": lambda x, t: (noop_space(x), noop_time(t)),
                       "temperature": lambda x, t: (noop_space(x), noop_time(t)),
                       "global_radiation": lambda x, t: (noop_space(x), noop_time(t)),
                       "precipitation": lambda x, t: (noop_space(x), noop_time(t)),
                       "precipitation_amount_acc": lambda x, t: (prec_acc_conv(x), dacc_time(t))}
        res = {}
        for k, (v, ak) in data.items():
            res[k] = convert_map[ak](v, time)
        return res

    def _geo_ts_to_vec(self, data, pts):
        res = {}
        for name, ts in iteritems(data):
            tpe = self.source_type_map[name]
            res[name] = tpe.vector_t([tpe(api.GeoPoint(*pts[idx]),
                                      ts[idx]) for idx in np.ndindex(pts.shape[:-1])])
        return res