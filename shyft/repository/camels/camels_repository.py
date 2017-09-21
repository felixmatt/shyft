from __future__ import absolute_import
from __future__ import print_function
from six import iteritems

import os
import glob
from os import path
import numpy as np
import pandas as pd
from pyproj import Proj
from pyproj import transform
from shyft import api
from shyft import shyftdata_dir
from shyft.repository import interfaces
from shyft.repository.service.ssa_geo_ts_repository import TsRepository
from shyft.orchestration.configuration.config_interfaces import RegionConfig, ModelConfig, RegionConfigError
from shyft.repository.netcdf.cf_region_model_repository import BoundingBoxRegion


class CamelsDataRepositoryError(Exception):
    pass


class CamelsDataRepository(interfaces.GeoTsRepository):
    """
    Repository for geo located timeseries stored in netCDF files.

    """
                     
    def __init__(self, epsg, path_to_database, sgid, selection_criteria=None):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP model,
        and initialize data retrieval.
        """

        self.sgid = sgid
        filename = self.get_filepath_from_sgid(path_to_database, sgid)
        self.selection_criteria = selection_criteria
        if not path.isfile(filename):
            raise CamelsDataRepositoryError("No such file '{}'".format(filename))
            
        self._filename = filename
        self.allow_subset = True  # allow_subset
        self.elevation_file = None

        self.shyft_cs = "+init=EPSG:{}".format(epsg)
        self._x_padding = 5000.0  # x_padding
        self._y_padding = 5000.0  # y_padding
        self._bounding_box = None  # bounding_box

        self._shift_fields = ()

        self.source_type_map = {"relative_humidity": api.RelHumSource,
                                "temperature": api.TemperatureSource,
                                "precipitation": api.PrecipitationSource,
                                "radiation": api.RadiationSource,
                                "wind_speed": api.WindSpeedSource}

        self.vector_type_map = {"discharge": api.TsVector}

        if self.selection_criteria is not None: self._validate_selection_criteria()

    @classmethod
    def get_filepath_from_sgid(self, CAMELS_DIR, sgid, streamflow=False):
        if not path.isabs(CAMELS_DIR):
            # Relative paths will be prepended the data_dir
            filename = path.join(shyftdata_dir, CAMELS_DIR)
        DATA_DIR = CAMELS_DIR+'/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2'
        met_forcing_type = 'daymet' # daymet | maurer | nldas
        TARGET_DIR = DATA_DIR+'/basin_mean_forcing/'+met_forcing_type
        ending = "_forcing_leap.txt"
        if streamflow:
            TARGET_DIR = DATA_DIR + '/usgs_streamflow'
            ending = "_streamflow_qc.txt"
        filepath = glob.glob(TARGET_DIR+'/*/'+sgid+'*'+ending)
        if len(filepath) != 1:
            raise CamelsDataRepositoryError("sgid {} is not unique in {}. Found files: {}".format(sgid, TARGET_DIR, filepath))
        return filepath[0]

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
            raise CamelsDataRepositoryError("File '{}' not found".format(filename))
        return self._get_data_from_camels_database(filename, input_source_types,
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

    def _validate_selection_criteria(self):
        s_c = self.selection_criteria
        if list(s_c)[0] == 'unique_id':
            if not isinstance(s_c['unique_id'], list):
                raise CamelsDataRepositoryError("Unique_id selection criteria should be a list.")
        elif list(s_c)[0] == 'polygon':
            raise CamelsDataRepositoryError("Selection using polygon not supported yet.")
        elif list(s_c)[0] == 'bbox':
            if not (isinstance(s_c['bbox'], list) and len(s_c['bbox']) == 2):
                raise CamelsDataRepositoryError("bbox selection criteria should be a list with two lists.")
            self._bounding_box = s_c['bbox']
        else:
            raise CamelsDataRepositoryError("Unrecognized selection criteria.")

    def _convert_to_timeseries(self, data_map):
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

        time_series = {}
        for key, (data, ta) in data_map.items():
            def construct(d):
                if ta.size() != d.size:
                    raise CFDataRepositoryError("Time axis size {} not equal to the number of "
                                                   "data points ({}) for {}"
                                                   "".format(ta.size(), d.size, key))
                return api.TimeSeries(ta, api.DoubleVector.FromNdArray(d), api.POINT_AVERAGE_VALUE)

            #time_series[key] = np.array([[construct(data[fslice + [i, j]])
            #                              for j in range(J)] for i in range(I)])
            time_series[key] = np.array([construct(data)])
        return time_series

    def _get_data_from_camels_database(self, filename, input_source_types, utc_period,
                               geo_location_criteria, ensemble_member=None):
        ts_id = None
        if self.selection_criteria is None:
            self.selection_criteria = {'bbox':geo_location_criteria}
            self._bounding_box = geo_location_criteria

        raw_data = {}
        x = 0.0 # TODO get x
        y = 0.0 # TODO get y

        station_info = {}
        with open(filename) as data:
            station_info['lat'] = float(data.readline())  # latitude of gauge
            station_info['z'] = float(data.readline())  # elevation of gauge (m)
            station_info['area'] = float(data.readline())  # area of basin (m^2)
        keys = ['Date', 'dayl', 'precipitation', 'radiation', 'swe', 'tmax', 'tmin', 'vp']
        data_dict = pd.read_table(filename, skiprows=4, names=keys)
        data_dict["temperature"] = (data_dict["tmin"] + data_dict["tmax"]) / 2.
        data_dict.pop("tmin")
        data_dict.pop("tmax")
        data_dict["wind_speed"] = data_dict["temperature"] * 0.0 + 2.0 # no wind speed in camels
        data_dict["relative_humidity"] = data_dict["temperature"] * 0.0 + 0.7 # no relative humidity in camels

        time = self._get_utc_time_from_daily_camels(data_dict['Date'].values)
        #data_cs = dataset.variables.get("crs", None)
        data_cs = 32633 #"epsg:1234" # TODO which crs?
        idx_min = np.searchsorted(time, utc_period.start, side='left')
        if time[idx_min] > utc_period.start and idx_min > 0:  # important ! ensure data *cover* the requested period, Shyft ts do take care of resolution etc.
            idx_min -= 1  # extend range downward so we cover the entire requested period
        idx_max = np.searchsorted(time, utc_period.end, side='right')
        if time[idx_max] < utc_period.end and idx_max + 1 < len(time):
            idx_max += 1  # extend range upward so that we cover the requested period
        time_slice = slice(idx_min, idx_max)

        for k in data_dict.keys():
            if k in input_source_types:
                #if k in self._shift_fields and issubset:  # Add one to time slice
                #    data_time_slice = slice(time_slice.start, time_slice.stop + 1)
                #else:
                data_time_slice = time_slice
                data = data_dict[k].values
                pure_arr = data[data_time_slice]
                if isinstance(pure_arr, np.ma.core.MaskedArray):
                    pure_arr = pure_arr.filled(np.nan)
                raw_data[k] = pure_arr
                #raw_data[self._nc_shyft_map[k]] = np.array(data[data_slice], dtype='d'), k

        pts = np.dstack(([0.0], [0.0], [0.0])).reshape(-1, 3) # TODO what's the location?
        self.pts = pts

        # Make sure requested fields are valid, and that dataset contains the requested data.
        if not self.allow_subset and not (set(raw_data.keys()).issuperset(input_source_types)):
            raise CamelsDataRepositoryError("Could not find all data fields")

        extracted_data = self._transform_raw(raw_data, time[time_slice])
        self.extracted_data = extracted_data
        return self._geo_ts_to_vec(self._convert_to_timeseries(extracted_data), pts)

    def _calculate_temperature(self, data_dict):
        pass

    def _add_missing_input(self, data_dict):
        pass

    def _get_utc_time_from_daily_camels(self, datestr_lst):
        utc = api.Calendar()
        time = [utc.time(*[int(i) for i in date.split(' ')]) for date in datestr_lst]
        return np.array(time)

    def _transform_raw(self, data, time):
        """
        We need full time if deaccumulating
        """

        def noop_time(t):
            return api.TimeAxis(api.UtcTimeVector.from_numpy(t.astype(int)), int(2*t[-1] - t[-2]))

        def noop_space(x):
            return x

        def prec_conv(p):
            return p/24. # mm/day -> mm/h

        # Unit- and aggregation-dependent conversions go here
        convert_map = {"wind_speed": lambda x, t: (noop_space(x), noop_time(t)),
                       "relative_humidity": lambda x, t: (noop_space(x), noop_time(t)),
                       "temperature": lambda x, t: (noop_space(x), noop_time(t)),
                       "radiation": lambda x, t: (noop_space(x), noop_time(t)),
                       "precipitation": lambda x, t: (prec_conv(x), noop_time(t)),
                       "discharge": lambda x, t: (noop_space(x), noop_time(t))}
        res = {}
        for k, v in data.items():
            res[k] = convert_map[k](v, time)
        return res

    def _geo_ts_to_vec(self, data, pts):
        res = {}
        for name, ts in iteritems(data):
            if name in self.source_type_map.keys():
                tpe = self.source_type_map[name]
                geo_ts_list = tpe.vector_t()
                for idx in np.ndindex(pts.shape[:-1]):
                    geo_ts = tpe(api.GeoPoint(*pts[idx]), ts[idx])
                    geo_ts_list.append(geo_ts)
                res[name] = geo_ts_list
            else:
                vct = self.vector_type_map[name]
                res[name] = vct(ts)
        return res


class CamelsTargetRepositoryError(Exception):
    pass


class CamelsTargetRepository(TsRepository):
    """
    Repository for geo located timeseries stored in netCDF files.

    """

    def __init__(self, path_to_database, sgid, var_type):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP model,
        and initialize data retrieval.
        """
        # directory = params['data_dir']
        self.sgid = sgid
        filename = CamelsDataRepository.get_filepath_from_sgid(path_to_database ,sgid, streamflow=True)
        self.var_name = var_type

        # if not path.isdir(directory):
        #    raise CFDataRepositoryError("No such directory '{}'".format(directory))
        if not path.isfile(filename):
            raise CamelsTargetRepositoryError("No such file '{}'".format(filename))
        self._filename = filename  # path.join(directory, filename)

    def read(self, list_of_ts_id, utc_period):
        if not utc_period.valid():
            raise CamelsTargetRepositoryError("period should be valid()  of type api.UtcPeriod")

        filename = self._filename

        if not path.isfile(filename):
            raise CamelsTargetRepositoryError("File '{}' not found".format(filename))
        return self._get_data_from_dataset(filename, utc_period, list_of_ts_id)

    def _convert_to_timeseries(self, data, t, ts_id):
        ta = api.TimeAxisFixedDeltaT(int(t[0]), int(t[1]) - int(t[0]), len(t))
        tsc = api.TsFactory().create_point_ts

        def construct(d):
            return tsc(ta.size(), ta.start, ta.delta_t,
                       api.DoubleVector.FromNdArray(d))

        ts = [construct(data[:])]
        return {k: v for k, v in zip(ts_id, ts)}

    def _get_data_from_dataset(self, filename, utc_period, ts_id_to_extract):
        #ts_id_key = [k for (k, v) in dataset.variables.items() if getattr(v, 'cf_role', None) == 'timeseries_id'][0]
        #ts_id_in_file = dataset.variables[ts_id_key][:]

        #time = dataset.variables.get("time", None)
        #data = dataset.variables.get(self.var_name, None)
        keys = ['sgid','Year','Month', 'Day', 'discharge', 'quality']
        data_dict = pd.read_table(filename, names=keys, delim_whitespace=True)
        time = self._get_utc_time_from_daily_camels_streamgauge(data_dict['Year'].values, data_dict['Month'].values, data_dict['Day'].values)
        idx_min = np.searchsorted(time, utc_period.start, side='left')
        if time[idx_min] > utc_period.start and idx_min > 0:  # important ! ensure data *cover* the requested period, Shyft ts do take care of resolution etc.
            idx_min -= 1  # extend range downward so we cover the entire requested period
        idx_max = np.searchsorted(time, utc_period.end, side='right')
        if time[idx_max] < utc_period.end and idx_max + 1 < len(time):
            idx_max += 1  # extend range upward so that we cover the requested period
        time_slice = slice(idx_min, idx_max)
        ts_id_in_file = ts_id_to_extract
        extracted_data = data_dict['discharge'].values[time_slice]
        missing = -999.0
        extracted_data[np.isclose(missing, extracted_data)] = np.nan # set missing values to nan
        extracted_data *= 0.0283168466 # cubic feet per sec to cubic meter per sec
        return self._convert_to_timeseries(extracted_data, time[time_slice], ts_id_in_file)

    def _get_utc_time_from_daily_camels_streamgauge(self, year, month, day):
        utc = api.Calendar()
        time = [utc.time(int(y), int(m), int(d), 12) for y,m,d in zip(year, month, day)]
        return np.array(time)


class CamelsRegionModelRepositoryError(Exception):
    pass


class CamelsRegionModelRepository(interfaces.RegionModelRepository):
    """
    Repository that delivers fully specified shyft api region_models
    based on data found in Camels database.
    """

    def __init__(self, region_config, model_config):
        """
        Parameters
        ----------
        region_config: subclass of interface RegionConfig
            Object containing regional information, like
            catchment overrides, and which netcdf file to read
        model_config: subclass of interface ModelConfig
            Object containing model information, i.e.
            information concerning interpolation and model
            parameters
        """
        if not isinstance(region_config, RegionConfig) or \
                not isinstance(model_config, ModelConfig):
            raise interfaces.InterfaceError()
        self._rconf = region_config
        self._mconf = model_config
        self._region_model = model_config.model_type()  # region_model
        self._mask = None
        self._epsg = self._rconf.domain()["EPSG"]  # epsg
        self.sgid = self._rconf.repository()["params"]["sgid"]
        path_to_database = self._rconf.repository()["params"]["path_to_database"]
        filename = CamelsDataRepository.get_filepath_from_sgid(path_to_database, self.sgid)
        if not path.isfile(filename):
            raise CamelsRegionModelRepositoryError("No such file '{}'".format(filename))
        self._data_file = filename
        self._catch_ids = self._rconf.catchments()
        self.bounding_box = None

    def get_region_model(self, region_id, catchments=None):
        """
        Return a fully specified shyft api region_model for region_id, based on data found
        in netcdf dataset.

        Parameters
        -----------
        region_id: string
            unique identifier of region in data

        catchments: list of unique integers
            catchment indices when extracting a region consisting of a subset
            of the catchments has attribs to construct params and cells etc.

        Returns
        -------
        region_model: shyft.api type
        """
        station_info = {}
        with open(self._data_file) as data:
            station_info['lat'] = float(data.readline())  # latitude of gauge
            station_info['z'] = float(data.readline())  # elevation of gauge (m)
            station_info['area'] = float(data.readline())  # area of basin (m^2)
            c_ids = np.array([int(self.sgid)])
            x = np.array([0.0])
            y = np.array([0.0])
            m_catch = np.ones(len(c_ids), dtype=bool)
            # if self._catch_ids is not None:
            #    m_catch = np.in1d(c_ids, self._catch_ids)
            #    xcoord_m = xcoord[m_catch]
            #    ycoord_m = ycoord[m_catch]

            dataset_epsg = 32633

            # if dataset_epsg != self._epsg:
            target_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                int(self._epsg) - 32600, "WGS84", "WGS84")
            source_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(
                int(dataset_epsg) - 32600, "WGS84", "WGS84")

            # Construct bounding region
            box_fields = set(("lower_left_x", "lower_left_y", "step_x", "step_y", "nx", "ny", "EPSG"))
            if box_fields.issubset(self._rconf.domain()):
                tmp = self._rconf.domain()
                epsg = tmp["EPSG"]
                x_min = tmp["lower_left_x"]
                x_max = x_min + tmp["nx"] * tmp["step_x"]
                y_min = tmp["lower_left_y"]
                y_max = y_min + tmp["ny"] * tmp["step_y"]
                bounding_region = BoundingBoxRegion(np.array([x_min, x_max]),
                                                    np.array([y_min, y_max]), epsg, self._epsg)
            else:
                bounding_region = BoundingBoxRegion(xcoord_m, ycoord_m, dataset_epsg, self._epsg)
            self.bounding_box = bounding_region.bounding_box(self._epsg)

            areas = [station_info['area']]
            elevation = [0.0]
            coordinates = np.dstack((x, y, elevation)).reshape(-1, 3)

            c_ids_unique = list(np.unique(c_ids))

            ff = np.array([0.0])
            lf = np.array([0.0])
            rf = np.array([0.0])
            gf = np.array([0.0])

        # Construct region parameter:
        name_map = {"priestley_taylor": "pt", "kirchner": "kirchner",
                    "precipitation_correction": "p_corr", "actual_evapotranspiration": "ae",
                    "gamma_snow": "gs", "skaugen_snow": "ss", "hbv_snow": "hs", "glacier_melt": "gm",
                    "hbv_actual_evapotranspiration": "ae", "hbv_soil": "soil", "hbv_tank": "tank",
                    "routing": "routing"}
        region_parameter = self._region_model.parameter_t()
        for p_type_name, value_ in iteritems(self._mconf.model_parameters()):
            if p_type_name in name_map:
                if hasattr(region_parameter, name_map[p_type_name]):
                    sub_param = getattr(region_parameter, name_map[p_type_name])
                    for p, v in iteritems(value_):
                        if hasattr(sub_param, p):
                            setattr(sub_param, p, v)
                        else:
                            raise RegionConfigError(
                                "Invalid parameter '{}' for parameter set '{}'".format(p, p_type_name))
                else:
                    raise RegionConfigError("Invalid parameter set '{}' for selected model '{}'".format(p_type_name,
                                                                                                        self._region_model.__name__))
            else:
                raise RegionConfigError("Unknown parameter set '{}'".format(p_type_name))

        radiation_slope_factor = 0.9  # TODO: Move into yaml file similar to p_corr_scale_factor
        unknown_fraction = 1.0 - gf - lf - rf - ff

        # Construct cells
        cell_geo_data = np.column_stack([x, y, elevation, areas, c_ids.astype(int), np.full(len(c_ids),
                                                                                            radiation_slope_factor),
                                         gf, lf, rf, ff, unknown_fraction])
        cell_vector = self._region_model.cell_t.vector_t.create_from_geo_cell_data_vector(np.ravel(cell_geo_data))

        # Construct catchment overrides
        catchment_parameters = self._region_model.parameter_t.map_t()
        for cid, catch_param in iteritems(self._rconf.parameter_overrides()):
            if cid in c_ids_unique:
                param = self._region_model.parameter_t(region_parameter)
                for p_type_name, value_ in iteritems(catch_param):
                    if p_type_name in name_map:
                        if hasattr(param, name_map[p_type_name]):
                            sub_param = getattr(param, name_map[p_type_name])
                            for p, v in iteritems(value_):
                                if hasattr(sub_param, p):
                                    setattr(sub_param, p, v)
                                else:
                                    raise RegionConfigError(
                                        "Invalid parameter '{}' for catchment parameter set '{}'".format(p,
                                                                                                         p_type_name))
                        else:
                            raise RegionConfigError(
                                "Invalid catchment parameter set '{}' for selected model '{}'".format(p_type_name,
                                                                                                      self._region_model.__name__))
                    else:
                        raise RegionConfigError("Unknown catchment parameter set '{}'".format(p_type_name))

                catchment_parameters[cid] = param
        region_model = self._region_model(cell_vector, region_parameter, catchment_parameters)
        region_model.bounding_region = bounding_region
        region_model.catchment_id_map = c_ids_unique
        return region_model