import os
from os import path
import numpy as np
from netCDF4 import Dataset
import pyproj
from shapely.ops import transform
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
from shapely.prepared import prep
from functools import partial
from shyft import api
from shyft import shyftdata_dir
from .. import interfaces
from .time_conversion import convert_netcdf_time

UTC = api.Calendar()

class ECConcatDataRepositoryError(Exception):
    pass


class ECConcatDataRepository(interfaces.GeoTsRepository):
    _G = 9.80665  # WMO-defined gravity constant to calculate the height in metres from geopotential

    # Constants used in RH calculation
    __a1_w=611.21 # Pa
    __a3_w=17.502
    __a4_w=32.198 # K

    __a1_i=611.21 # Pa
    __a3_i=22.587
    __a4_i=-20.7 # K

    __T0=273.16 # K
    __Tice=205.16 # K

    def __init__(self, epsg, filename, nb_pads=0, nb_fc_to_drop=0, selection_criteria=None, padding=5000.):
        self.selection_criteria = selection_criteria
        # filename = filename.replace('${SHYFTDATA}', os.getenv('SHYFTDATA', '.'))
        filename = path.expandvars(filename)
        if not path.isabs(filename):
            # Relative paths will be prepended the data_dir
            filename = path.join(shyftdata_dir, filename)
        if not path.isfile(filename):
            raise ECConcatDataRepositoryError("No such file '{}'".format(filename))

        self._filename = filename
        self.nb_pads = nb_pads
        self.nb_fc_to_drop = nb_fc_to_drop  # index of first lead time: starts from 0
        self.nb_fc_interval_to_concat = 1  # given as number of forecast intervals
        self.shyft_cs = "+init=EPSG:{}".format(epsg)
        self.padding = padding
        
        # Field names and mappings netcdf_name: shyft_name
        self._arome_shyft_map = {'dew_point_temperature_2m': 'dew_point_temperature_2m',
                                 'surface_air_pressure': 'surface_air_pressure',
                                 #"relative_humidity_2m": "relative_humidity",
                                 "air_temperature_2m": "temperature",
                                 #"precipitation_amount": "precipitation",
                                 "precipitation_amount_acc": "precipitation",
                                 "x_wind_10m": "x_wind",
                                 "y_wind_10m": "y_wind",
                                 "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time": "radiation"}

        self.var_units = {'dew_point_temperature_2m': ['K'],
                          'surface_air_pressure': ['Pa'],
                          "air_temperature_2m": ['K'],
                          "precipitation_amount_acc": ['kg/m^2'],
                          "x_wind_10m": ['m/s'],
                          "y_wind_10m": ['m/s'],
                          "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time": ['W s/m^2']}

        self._shift_fields = ("precipitation_amount_acc",
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

        if self.selection_criteria is not None: self._validate_selection_criteria()
        
    def _validate_selection_criteria(self):
        s_c = self.selection_criteria
        if list(s_c)[0] == 'unique_id':
            if not isinstance(s_c['unique_id'], list):
                raise ECConcatDataRepositoryError("Unique_id selection criteria should be a list.")
        elif list(s_c)[0] == 'polygon':
            if not isinstance(s_c['polygon'], (Polygon, MultiPolygon)):
                raise ECConcatDataRepositoryError(
                        "polygon selection criteria should be one of these shapley objects: (Polygon, MultiPolygon).")
        elif list(s_c)[0] == 'bbox':
            if not (isinstance(s_c['bbox'], tuple) and len(s_c['bbox']) == 2):
                raise ECConcatDataRepositoryError("bbox selection criteria should be a tuple with two numpy arrays.")
            self._bounding_box = s_c['bbox']
        else:
            raise ECConcatDataRepositoryError("Unrecognized selection criteria.")

    def get_timeseries(self, input_source_types, utc_period, geo_location_criteria=None):
        """Get shyft source vectors of time series for input_source_types
        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve (precipitation, temperature..)
        geo_location_criteria: object, optional
            bbox or shapely polygon
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.
        Returns
        -------
        geo_loc_ts: dictionary
            dictionary keyed by time series name, where values are api vectors of geo
            located timeseries.
        """

        with Dataset(self._filename) as dataset:
            return self._get_data_from_dataset(dataset, input_source_types,
                                               utc_period, geo_location_criteria, concat=True)

    def get_forecasts(self, input_source_types, fc_selection_criteria, geo_location_criteria):
        k, v = list(fc_selection_criteria.items())[0]
        if k == 'forecasts_within_period':
            if not isinstance(v, api.UtcPeriod):
                raise ECConcatDataRepositoryError("'forecasts_within_period' selection criteria should be of type api.UtcPeriod.")
        elif k == 'latest_available_forecasts':
            if not all([isinstance(v, dict), isinstance(v['number of forecasts'], int), isinstance(v['forecasts_older_than'], int)]):
                raise ECConcatDataRepositoryError("'latest_available_forecasts' selection criteria should be of type dict.")
        elif k == 'forecasts_at_reference_times':
            if not isinstance(v, list):
                raise ECConcatDataRepositoryError("'forecasts_at_reference_times' selection criteria should be of type list.")
        else:
            raise ECConcatDataRepositoryError("Unrecognized forecast selection criteria.")

        with Dataset(self._filename) as dataset:
            return self._get_data_from_dataset(dataset, input_source_types,
                                               v, geo_location_criteria, concat=False)

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

    def _convert_to_timeseries(self, data, concat):
        """Convert timeseries from numpy structures to shyft.api timeseries.
        Returns
        -------
        timeseries: dict
            Time series arrays keyed by type
        """
        tsf = api.TsFactory().create_point_ts
        tsc = api.TimeSeries
        time_series = {}
        if concat:
            for key, (data, ta) in data.items():
                nb_timesteps, nb_pts = data.shape

                def construct(d):
                    if ta.size() != d.size:
                        raise ECConcatDataRepositoryError("Time axis size {} not equal to the number of "
                                                       "data points ({}) for {}"
                                                       "".format(ta.size(), d.size, key))
                    return tsf(ta.size(), ta.start, ta.delta_t,
                               api.DoubleVector.FromNdArray(d.flatten()), self.series_type[key])
                time_series[key] = np.array([construct(data[:, j]) for j in range(nb_pts)])
        else:
            def construct(d, tax):
                if tax.size() != d.size:
                    raise ECConcatDataRepositoryError("Time axis size {} not equal to the number of "
                                                         "data points ({}) for {}"
                                                         "".format(tax.size(), d.size, key))
                return tsc(tax, api.DoubleVector.FromNdArray(d.flatten()), self.series_type[key])
            for key, (data, ta) in data.items():
                nb_forecasts, nb_timesteps, nb_pts = data.shape
                time_series[key] = np.array([construct(data[i, :, j], ta[i]) for i in range(nb_forecasts) for j in range(nb_pts)])
        return time_series

    def _limit(self, x, y, data_cs, target_cs, ts_id):
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
        data_proj = pyproj.Proj(data_cs)
        target_proj = pyproj.Proj(target_cs)

        if(list(self.selection_criteria)[0]=='bbox'):
            # Find bounding box in netcdf projection
            bbox = np.array(self.selection_criteria['bbox'])
            bbox[0][0] -= self.padding
            bbox[0][1] += self.padding
            bbox[0][2] += self.padding
            bbox[0][3] -= self.padding
            bbox[1][0] -= self.padding
            bbox[1][1] -= self.padding
            bbox[1][2] += self.padding
            bbox[1][3] += self.padding
            bb_proj = pyproj.transform(target_proj, data_proj, bbox[0], bbox[1])
            x_min, x_max = min(bb_proj[0]), max(bb_proj[0])
            y_min, y_max = min(bb_proj[1]), max(bb_proj[1])

            # Limit data
            xy_mask = ((x <= x_max) & (x >= x_min) & (y <= y_max) & (y >= y_min))

        if (list(self.selection_criteria)[0] == 'polygon'):
            poly = self.selection_criteria['polygon']
            pts_in_file = MultiPoint(np.dstack((x,y)).reshape(-1, 2))
            project = partial(pyproj.transform, target_proj, data_proj)
            poly_prj = transform(project, poly)
            p_poly = prep(poly_prj.buffer(self.padding))
            xy_mask = np.array(list(map(p_poly.contains, pts_in_file)))

        if(list(self.selection_criteria)[0]=='unique_id'):
            xy_mask = np.array([id in self.selection_criteria['unique_id'] for id in ts_id])

        # Check if there is at least one point extaracted and raise error if there isn't
        if not xy_mask.any():
            raise ECConcatDataRepositoryError("No points in dataset which satisfy selection criterion '{}'.".
                                              format(list(self.selection_criteria)[0]))

        xy_inds = np.nonzero(xy_mask)[0]

        # Transform from source coordinates to target coordinates
        xx, yy = pyproj.transform(data_proj, target_proj, x[xy_mask], y[xy_mask])

        return xx, yy, xy_mask, slice(xy_inds[0], xy_inds[-1] + 1)

    def _make_time_slice(self, time, lead_time, lead_times_in_sec, fc_selection_criteria_v, concat):
        v = fc_selection_criteria_v
        nb_extra_intervals = 0
        if concat: # make continuous timeseries
            self.fc_len_to_concat = self.nb_fc_interval_to_concat * self.fc_interval
            utc_period = v  # TODO: verify that fc_selection_criteria_v is of type api.UtcPeriod
            time_after_drop = time + lead_times_in_sec[self.nb_fc_to_drop]
            # idx_min = np.searchsorted(time, utc_period.start, side='left')
            idx_min = np.argmin(time_after_drop <= utc_period.start) - 1  # raise error if result is -1
            #idx_max = np.searchsorted(time, utc_period.end, side='right')
            idx_max = np.argmax(time_after_drop >= utc_period.end)  # raise error if result is 0
            if idx_min<0:
                first_lead_time_of_last_fc = int(time_after_drop[-1])
                if first_lead_time_of_last_fc <= utc_period.start:
                    idx_min = len(time)-1
                else:
                    raise ECConcatDataRepositoryError(
                        "The earliest time in repository ({}) is later than the start of the period for which data is "
                        "requested ({})".format(UTC.to_string(int(time_after_drop[0])), UTC.to_string(utc_period.start)))
            if idx_max == 0:
                last_lead_time_of_last_fc = int(time[-1] + lead_times_in_sec[-1])
                if last_lead_time_of_last_fc < utc_period.end:
                    raise ECConcatDataRepositoryError(
                        "The latest time in repository ({}) is earlier than the end of the period for which data is "
                        "requested ({})".format(UTC.to_string(last_lead_time_of_last_fc), UTC.to_string(utc_period.end)))
                else:
                    idx_max = len(time)-1

            #issubset = True if idx_max < len(time) - 1 else False # For a concat repo 'issubset' is related to the lead_time axis and not the main time axis
            issubset = True if self.nb_fc_to_drop + self.fc_len_to_concat < len(lead_time)-1 else False
            time_slice = slice(idx_min, idx_max+1)
            last_time = int(time[idx_max]+lead_times_in_sec[self.nb_fc_to_drop + self.fc_len_to_concat - 1])
            if utc_period.end > last_time:
                nb_extra_intervals = int(0.5+(utc_period.end-last_time)/(self.fc_len_to_concat*self.fc_time_res))
        else:
            #self.fc_len_to_concat = len(lead_time)  # Take all lead_times for now
            #self.nb_fc_to_drop = 0  # Take all lead_times for now
            self.fc_len_to_concat = len(lead_time) - self.nb_fc_to_drop
            if isinstance(v, api.UtcPeriod):
                time_slice = ((time >= v.start) & (time <= v.end))
                if not any(time_slice):
                    raise ECConcatDataRepositoryError(
                        "No forecasts found with start time within period {}.".format(v.to_string()))
            elif isinstance(v, list):
                raise ECConcatDataRepositoryError(
                        "'forecasts_at_reference_times' selection criteria not supported yet.")
            elif isinstance(v, dict):  # get the latest forecasts
                t = v['forecasts_older_than']
                n = v['number of forecasts']
                idx = np.argmin(time <= t) - 1
                if idx < 0:
                    first_lead_time_of_last_fc = int(time[-1])
                    if first_lead_time_of_last_fc < t:
                        idx = len(time) - 1
                    else:
                        raise ECConcatDataRepositoryError(
                            "The earliest time in repository ({}) is later than or at the start of the period for which data is "
                            "requested ({})".format(UTC.to_string(int(time[0])), UTC.to_string(t)))
                if idx+1 < n:
                    raise ECConcatDataRepositoryError(
                        "The number of forecasts available in repo ({}) and earlier than the parameter "
                        "'forecasts_older_than' ({}) is less than the number of forecasts requested ({})".format(
                            idx+1, UTC.to_string(t), n))
                time_slice = slice(idx-n+1, idx+1)
            issubset = False  # Since we take all the lead_times for now
        lead_time_slice = slice(self.nb_fc_to_drop, self.nb_fc_to_drop + self.fc_len_to_concat)

        #For checking
        # print('Time slice:', UTC.to_string(int(time[time_slice][0])), UTC.to_string(int(time[time_slice][-1])))

        return time_slice, lead_time_slice, issubset, self.fc_len_to_concat, nb_extra_intervals

    def _get_data_from_dataset(self, dataset, input_source_types, fc_selection_criteria_v,
                               geo_location_criteria, concat=True, ensemble_member=None):
        ts_id = None
        if geo_location_criteria is not None:
            self.selection_criteria = geo_location_criteria
        self._validate_selection_criteria()
        if list(self.selection_criteria)[0]=='unique_id':
            ts_id_key = [k for (k, v) in dataset.variables.items() if getattr(v, 'cf_role', None) == 'timeseries_id'][0]
            ts_id = dataset.variables[ts_id_key][:]

        if "wind_speed" in input_source_types:
            input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("wind_speed")
            input_source_types.append("x_wind")
            input_source_types.append("y_wind")
        no_temp = False
        if "temperature" not in input_source_types: no_temp = True
        if "relative_humidity" in input_source_types:
            if not isinstance(input_source_types, list):
                input_source_types = list(input_source_types)  # We change input list, so take a copy
            input_source_types.remove("relative_humidity")
            input_source_types.extend(["surface_air_pressure", "dew_point_temperature_2m"])
            if no_temp: input_source_types.extend(["temperature"])

        unit_ok = {k:dataset.variables[k].units in self.var_units[k]
                      for k in dataset.variables.keys() if self._arome_shyft_map.get(k, None) in input_source_types}
        if not all(unit_ok.values()):
            raise ECConcatDataRepositoryError("The following variables have wrong unit: {}.".format(
                ', '.join([k for k, v in unit_ok.items() if not v])))

        raw_data = {}
        x = dataset.variables.get("x", None)
        y = dataset.variables.get("y", None)
        time = dataset.variables.get("time", None)
        lead_time = dataset.variables.get("lead_time", None)
        dim_nb_series = [dim.name for dim in dataset.dimensions.values() if dim.name not in ['time', 'lead_time']][0]
        if not all([x, y, time, lead_time]):
            raise ECConcatDataRepositoryError("Something is wrong with the dataset."
                                           " x/y coords or time not found.")
        data_cs = dataset.variables.get("crs", None)
        if data_cs is None:
            raise ECConcatDataRepositoryError("No coordinate system information in dataset.")

        time = convert_netcdf_time(time.units,time)
        lead_times_in_sec = lead_time[:]*3600.
        self.fc_time_res = (lead_time[1]-lead_time[0])*3600. # in seconds
        self.fc_interval = int((time[1]-time[0])/self.fc_time_res)  # in-terms of self.fc_time_res

        time_slice, lead_time_slice, issubset, self.fc_len_to_concat, nb_extra_intervals = \
            self._make_time_slice(time, lead_time, lead_times_in_sec,fc_selection_criteria_v, concat)

        time_ext = time[time_slice]
        # print('nb_extra_intervals:',nb_extra_intervals)
        if nb_extra_intervals > 0:
            time_extra = time_ext[-1]+np.arange(1, nb_extra_intervals+1)*self.fc_len_to_concat*self.fc_time_res
            time_ext = np.concatenate((time_ext, time_extra))
            # print('Extra time:', time_ext)

        x, y, m_xy, xy_slice = self._limit(x[:], y[:], data_cs.proj4, self.shyft_cs, ts_id)
        for k in dataset.variables.keys():
            if self._arome_shyft_map.get(k, None) in input_source_types:

                if k in self._shift_fields and issubset:  # Add one to lead_time slice
                    data_lead_time_slice = slice(lead_time_slice.start, lead_time_slice.stop + 1)
                else:
                    data_lead_time_slice = lead_time_slice

                data = dataset.variables[k]
                dims = data.dimensions
                data_slice = len(data.dimensions)*[slice(None)]
                if ensemble_member is not None:
                    data_slice[dims.index("ensemble_member")] = ensemble_member
                data_slice[dims.index(dim_nb_series)] = xy_slice
                data_slice[dims.index("lead_time")] = data_lead_time_slice
                data_slice[dims.index("time")] = time_slice # data_time_slice
                new_slice = [m_xy[xy_slice] if dim==dim_nb_series else slice(None) for dim in dims ]
                pure_arr = data[data_slice][new_slice]
                # To check equality of the two extraction methods
                # data_slice[dims.index(dim_nb_series)] = m_xy
                # print('Diff:', np.sum(data[data_slice]-pure_arr)) # This should be 0.0
                if isinstance(pure_arr, np.ma.core.MaskedArray):
                    pure_arr = pure_arr.filled(np.nan)
                if nb_extra_intervals > 0:
                    data_slice[dims.index("time")] = [time_slice.stop - 1]
                    data_slice[dims.index("lead_time")] = slice(data_lead_time_slice.stop,
                                                                data_lead_time_slice.stop + (nb_extra_intervals+1) * self.fc_len_to_concat)
                    data_extra = data[data_slice][new_slice].reshape(nb_extra_intervals+1, self.fc_len_to_concat, -1)
                    if k in self._shift_fields:
                        data_extra_ = np.zeros((nb_extra_intervals, self.fc_len_to_concat+1, len(x)), dtype=data_extra.dtype)
                        data_extra_[:, 0:-1, :] = data_extra[:-1, :, :]
                        data_extra_[:, -1, :] = data_extra[1:, -1, :]
                        data_extra = data_extra_
                    else:
                        data_extra = data_extra[:-1]
                    # print('Extra data shape:', data_extra.shape)
                    # print('Main data shape:', pure_arr.shape)
                    raw_data[self._arome_shyft_map[k]] = np.concatenate((pure_arr, data_extra)), k
                else:
                    raw_data[self._arome_shyft_map[k]] = pure_arr, k

        if 'z' in dataset.variables.keys():
            data = dataset.variables['z']
            dims = data.dimensions
            data_slice = len(data.dimensions) * [slice(None)]
            data_slice[dims.index(dim_nb_series)] = m_xy
            z = data[data_slice]
        else:
            raise ECConcatDataRepositoryError("No elevations found in dataset")

        pts = np.dstack((x, y, z)).reshape(-1,3)
        if not concat:
            pts = np.tile(pts, (len(time[time_slice]), 1))
        self.pts = pts

        if set(("x_wind", "y_wind")).issubset(raw_data):
            x_wind, _ = raw_data.pop("x_wind")
            y_wind, _ = raw_data.pop("y_wind")
            raw_data["wind_speed"] = np.sqrt(np.square(x_wind) + np.square(y_wind)), "wind_speed"
        if set(("surface_air_pressure", "dew_point_temperature_2m")).issubset(raw_data):
            sfc_p, _ = raw_data.pop("surface_air_pressure")
            dpt_t, _ = raw_data.pop("dew_point_temperature_2m")
            if no_temp:
                sfc_t, _ = raw_data.pop("temperature")
            else:
                sfc_t, _ = raw_data["temperature"]
            raw_data["relative_humidity"] = self.calc_RH(sfc_t, dpt_t, sfc_p), "relative_humidity"
        data_lead_time_slice = slice(lead_time_slice.start, lead_time_slice.stop + 1)
        extracted_data = self._transform_raw(raw_data, time_ext, lead_times_in_sec[data_lead_time_slice], concat)
        #self.extracted_data = extracted_data
        return self._geo_ts_to_vec(self._convert_to_timeseries(extracted_data, concat), pts)

    def _transform_raw(self, data, time, lead_time, concat):
        """
        We need full time if deaccumulating
        """

        def concat_t(t):
            t_stretch = np.ravel(np.repeat(t, self.fc_len_to_concat).reshape(len(t), self.fc_len_to_concat) + lead_time[0:self.fc_len_to_concat])
            return api.TimeAxisFixedDeltaT(int(t_stretch[0]), int(t_stretch[1]) - int(t_stretch[0]), len(t_stretch))

        def forecast_t(t, daccumulated_var=False):
            nb_ext_lead_times = self.fc_len_to_concat - 1 if daccumulated_var else self.fc_len_to_concat
            t_all = np.repeat(t, nb_ext_lead_times).reshape(len(t), nb_ext_lead_times) + lead_time[0:nb_ext_lead_times]
            return t_all.astype(int)

        def pad(v, t):
            if not concat:
                if self.nb_pads>0:
                    nb_pads = self.nb_pads
                    t_padded = np.zeros((t.shape[0],t.shape[1]+nb_pads), dtype=t.dtype)
                    t_padded[:,:-nb_pads] = t[:,:]
                    t_add = t[0,-1] - t[0,-nb_pads-1]
                    # print('t_add:',t_add)
                    t_padded[:,-nb_pads:] = t[:,-nb_pads:] + t_add

                    v_padded = np.zeros((v.shape[0],t.shape[1]+nb_pads,v.shape[2]), dtype=v.dtype)
                    v_padded[:, :-nb_pads, :] = v[:, :, :]
                    v_padded[:, -nb_pads:, :] = v[:, -nb_pads:, :]

                else:
                    t_padded = t
                    v_padded = v
                dt_last = t_padded[0, -1] - t_padded[0, -2]
                return (v_padded,
                        [api.TimeAxis(api.UtcTimeVector.from_numpy(t_one), int(t_one[-1] + dt_last)) for t_one in
                         t_padded])
            else:
                return (v, t)

        def concat_v(x):
            return x.reshape(-1, x.shape[-1])  # shape = (nb_forecasts*nb_lead_times, nb_points)

        def forecast_v(x):
            return x  # shape = (nb_forecasts, nb_lead_times, nb_points)

        def air_temp_conv(T, fcn):
            return fcn(T - 273.15)

        def prec_acc_conv(p, fcn):
            f = api.deltahours(1) / (lead_time[1:]-lead_time[:-1]) # conversion from mm/delta_t to mm/1hour
            return fcn(np.clip((p[:, 1:, :] - p[:, :-1, :])*f[np.newaxis,:,np.newaxis], 0.0, 1000.0))

        def rad_conv(r, fcn):
            dr = r[:, 1:, :] - r[:, :-1, :]
            return fcn(np.clip(dr / (lead_time[1:]-lead_time[:-1])[np.newaxis,:,np.newaxis], 0.0, 5000.0))

        # Unit- and aggregation-dependent conversions go here
        if concat:
            convert_map = {"wind_speed": lambda x, t: (concat_v(x), concat_t(t)),
                           "relative_humidity": lambda x, t: (concat_v(x), concat_t(t)),
                           "air_temperature_2m": lambda x, t: (air_temp_conv(x, concat_v), concat_t(t)),
                           "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time":
                               lambda x, t: (rad_conv(x, concat_v), concat_t(t)),
                           #"precipitation_amount": lambda x, t: (prec_conv(x), dacc_time(t)),
                           "precipitation_amount_acc": lambda x, t: (prec_acc_conv(x, concat_v), concat_t(t))}
        else:
            convert_map = {"wind_speed": lambda x, t: (forecast_v(x), forecast_t(t)),
                           "relative_humidity": lambda x, t: (forecast_v(x), forecast_t(t)),
                           "air_temperature_2m": lambda x, t: (air_temp_conv(x, forecast_v), forecast_t(t)),
                           "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time":
                               lambda x, t: (rad_conv(x, forecast_v), forecast_t(t, True)),
                           # "precipitation_amount": lambda x, t: (prec_conv(x), dacc_time(t)),
                           "precipitation_amount_acc": lambda x, t: (prec_acc_conv(x, forecast_v), forecast_t(t, True))}
        res = {}
        for k, (v, ak) in data.items():
            res[k] = pad(*convert_map[ak](v, time))
        return res

    def _geo_ts_to_vec(self, data, pts):
        res = {}
        for name, ts in data.items():
            tpe = self.source_type_map[name]
            tpe_v = tpe.vector_t()
            for idx in np.ndindex(pts.shape[:-1]):
                tpe_v.append(tpe(api.GeoPoint(*pts[idx]), ts[idx]))
            res[name] = tpe_v
        return res

    @classmethod
    def calc_q(cls, T, p, alpha):
        e_w = cls.__a1_w * np.exp(cls.__a3_w * ((T - cls.__T0) / (T - cls.__a4_w)))
        e_i = cls.__a1_i * np.exp(cls.__a3_i * ((T - cls.__T0) / (T - cls.__a4_i)))
        q_w = 0.622 * e_w / (p - (1 - 0.622) * e_w)
        q_i = 0.622 * e_i / (p - (1 - 0.622) * e_i)
        return alpha * q_w + (1 - alpha) * q_i

    @classmethod
    def calc_alpha(cls, T):
        alpha = np.zeros(T.shape, dtype='float')
        # alpha[T<=Tice]=0.
        alpha[T >= cls.__T0] = 1.
        indx = (T < cls.__T0) & (T > cls.__Tice)
        alpha[indx] = np.square((T[indx] - cls.__Tice) / (cls.__T0 - cls.__Tice))
        return alpha

    @classmethod
    def calc_RH(cls, T, Td, p):
        alpha = cls.calc_alpha(T)
        qsat = cls.calc_q(T, p, alpha)
        q = cls.calc_q(Td, p, alpha)
        return q / qsat