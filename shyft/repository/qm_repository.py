from . import interfaces
from shyft import api
import numpy as np

class QMRepositoryError(Exception):
    pass

class QMRepository(interfaces.GeoTsRepository):

    def __init__(self, qm_cfg_params, qm_resolution, repo_prior_idx, qm_interp_hours):
        # TODO: Check input
        self.qm_cfg_params = qm_cfg_params
        self.qm_resolution = qm_resolution
        self.repo_prior_idx = repo_prior_idx
        self.qm_interp_hours = qm_interp_hours
        self.source_type_map = {"relative_humidity": api.RelHumSource,
                                "temperature": api.TemperatureSource,
                                "precipitation": api.PrecipitationSource,
                                "radiation": api.RadiationSource,
                                "wind_speed": api.WindSpeedSource}
        self.source_vector_map = {"relative_humidity": api.RelHumSourceVector,
                                "temperature": api.TemperatureSourceVector,
                                "precipitation": api.PrecipitationSourceVector,
                                "radiation": api.RadiationSourceVector,
                                "wind_speed": api.WindSpeedSourceVector}

    def _read_fcst(self, start_time, input_source_types, bbox):
        # Read forecasts from repository for each forecast group
        # Hierachy of return is list of forecast class, list of forecasts, list of forecast members where each member
        # is a source-type keyed dictionary of geo time seies. Within each forecast class the newest member is last in
        # class
        raw_fcst_lst = []
        for i in range(len(self.qm_cfg_params)):
            nb_of_fcst = len(self.qm_cfg_params[i]['w'])
            is_ens = self.qm_cfg_params[i]['ens']
            if is_ens:
                # TODO: revise this when get_forecast_ensembles is ready in concat versions
                # raw_fcst = self.qm_cfg_params[i]['repo'].get_forecast_ensemble(self.start_time, self.bbox)
                # we read only one ensemble for now until ec_concat for ensemble is ready
                # when we have get_forecast_ensembles (read multiple ensemble sets at once) method available
                # in arome_concat and ec_concat OR arome raw and ec raw then we use that
                raw_fcst = self.qm_cfg_params[i]['repo'].get_forecast_ensemble(input_source_types, None, start_time,
                                                                               geo_location_criteria=bbox)
            else:
                # TODO: revise this when get_forecast_ensembles is ready in concat versions
                # if using either arome_concat or ec_concat return as list to match output from get_forecast_ensemble
                raw_fcst = [self.qm_cfg_params[i]['repo'].get_forecasts(input_source_types,
                               {'latest_available_forecasts': {'number of forecasts': nb_of_fcst,
                                                               'forecasts_older_than': start_time}},
                               geo_location_criteria={'bbox':bbox})]

                # if using arome raw and ec raw then we either need to call get_forecast two times OR
                # we need to make a get_forecasts method which internally fetches from multiples files

            # Examine first input_source_type to determine number of geo_points and use this to slice into raw_fcst to
            # return one list entry per forecast member
            fc_start_t, nb_geo_pts = np.unique([src.ts.time(0) for src in raw_fcst[0][input_source_types[0]]],
                                           return_counts=True)
            nb_geo_pts = int(nb_geo_pts[0])
            slices = [slice(i * nb_geo_pts, (i + 1) * nb_geo_pts) for i in range(nb_of_fcst)]
            forecasts = [[{src_type: memb[src_type][slc] for src_type in input_source_types} for memb in raw_fcst] for
                         slc in slices]
            raw_fcst_lst.append(forecasts)
        return raw_fcst_lst

    def _get_geo_pts(self, raw_fcst):
        # extract geo_pts from first souce vector and return as GeoPointVector
        first_fcst_member = raw_fcst[0][0]
        first_src_vec = first_fcst_member[list(first_fcst_member.keys())[0]]
        start_times, nb_geo_pts = np.unique([src.ts.time(0) for src in first_src_vec],
                                            return_counts=True)
        nb_geo_pts = nb_geo_pts[0]
        gpv = api.GeoPointVector()
        for k in range(0, nb_geo_pts):
            gpv.append(first_src_vec[k].mid_point())
        return gpv

    def _read_prior(self):
        # to get the last (latest) forecast of multiple fcsts
        pass

    def _prep_prior(self):
        pass

    def _downscaling(self, forecast, target_grid, ta_fixed_dt):
        # Using idw for time being
        prep_fcst = {}
        for src_type, fcst in forecast.items():
            if src_type == 'precipitation':
                # just setting som idw_params for time being
                idw_params = api.IDWPrecipitationParameter()
                idw_params.max_distance = 15000
                idw_params.max_members = 4
                idw_params.scale_factor = 1.0
                prep_fcst[src_type] = api.idw_precipitation(fcst, target_grid, ta_fixed_dt, idw_params)
            elif src_type == 'temperature':
                # just setting som idw_params for time being
                idw_params = api.IDWTemperatureParameter()
                idw_params.max_distance = 15000
                idw_params.max_members = 4
                idw_params.gradient_by_equation = False
                prep_fcst[src_type] = api.idw_temperature(fcst, target_grid, ta_fixed_dt, idw_params)
            elif src_type == 'radiation':
                # just setting som idw_params for time being
                idw_params = api.IDWParameter()
                idw_params.max_distance = 15000
                idw_params.max_members = 4
                idw_params.distance_measure_factor = 1
                slope_factor = api.DoubleVector([0.9]*len(target_grid))
                prep_fcst[src_type] = api.idw_radiation(fcst, target_grid, ta_fixed_dt, idw_params, slope_factor)
            elif src_type == 'wind_speed':
                # just setting som idw_params for time being
                idw_params = api.IDWParameter()
                idw_params.max_distance = 15000
                idw_params.max_members = 4
                idw_params.distance_measure_factor = 1
                prep_fcst[src_type] = api.idw_wind_speed(fcst, target_grid, ta_fixed_dt, idw_params)
            elif src_type == 'relative_humidity':
                # just setting som idw_params for time being
                idw_params = api.IDWParameter()
                idw_params.max_distance = 15000
                idw_params.max_members = 4
                idw_params.distance_measure_factor = 1
                prep_fcst[src_type] = api.idw_relative_humidity(fcst, target_grid, ta_fixed_dt, idw_params)

        return prep_fcst

    def _reduce_fcst_group_horizon(self, fcst_group, nb_hours):
        # for each fcst in group; create time axis for clipping
        clipped_fcst_group = []
        for fcst in fcst_group:
            # Get time acces from first src type in first member
            ta = fcst[0][list(fcst[0].keys())[0]][0].ts.time_axis
            clip_end = ta.time(0) + nb_hours * api.deltahours(1)
            if ta.time(0) < clip_end < ta.total_period().end:
                if ta.timeaxis_type == api.TimeAxisType.FIXED:
                    dt = ta.time(1) - ta.time(0)
                    n = nb_hours * api.deltahours(1) // dt
                    ta = api.TimeAxis(ta.time(0), dt, n)
                else:
                    idx = ta.time_points < clip_end
                    t_end = ta.time(int(idx.nonzero()[0][-1] + 1))
                    ta = api.TimeAxis(api.UtcTimeVector(ta.time_points[idx].tolist()), t_end)
            clipped_fcst_group.append(self._clip_forecast(fcst, ta))
        return clipped_fcst_group

    def _clip_forecast(self, fcst_ens, ta):
        # Clip ensemble forecast with ta
        new_fcst_ens = []
        for fcst in fcst_ens:
            src_dict = {}
            for src, geo_ts in fcst.items():
                vct = self.source_vector_map[src]()
                [vct.append(self.source_type_map[src](obj.mid_point(), obj.ts.average(ta))) for obj in geo_ts]
                src_dict[src] = vct
            new_fcst_ens.append(src_dict)
        return new_fcst_ens

    def _prep_fcst(self, start_time, raw_fcst_lst, qm_resolution_idx, ta):
        # Identifies space-time resolution and calls downscaling routine

        qm_cfg_params = self.qm_cfg_params
        # qm_resolution_idx, ta = self.qm_resolution[resolution_key]

        # Use prior resolution as target resolution if nothing is specified
        if qm_resolution_idx is None and repo_prior_idx is not None:
            qm_resolution_idx = repo_prior_idx
        if qm_resolution_idx is not None:
            target_grid = self._get_geo_pts(raw_fcst_lst[qm_resolution_idx])
        # TODO: if qm_resolution_idx is None and repo_prior_idx is None read prior and use as target grid

        prep_fcst_lst = []
        for i in range(len(qm_cfg_params)):
            raw_fcst_group = raw_fcst_lst[i]
            nb_hours = qm_cfg_params[i]['nb_hours']
            if nb_hours is not None:
                raw_fcst_group = self._reduce_fcst_group_horizon(raw_fcst_group, nb_hours)

            # TODO: is downscaling required? If target_grid is obtained from fcst or all values are nan there is no need to downscale.
            # Changing time axis resolution might still be required or can quantile_map_forecast handle this?
            prep_fcst = [[self._downscaling(f_m, target_grid, ta.fixed_dt) for f_m in fct]
                         for fct in raw_fcst_group]

            prep_fcst_lst.append(prep_fcst)

        return prep_fcst_lst, target_grid

    def _call_qm(self, prep_fcst_lst, weights, geo_points, ta, input_source_types, nb_prior_scenarios):

        # Check interpolation period is within time axis (ta)
        ta_start = ta.time(0)
        ta_end = ta.time(ta.size()-1) # start of last time step
        interp_start = ta_start + api.deltahours(self.qm_interp_hours[0])
        interp_end = ta_start + api.deltahours(self.qm_interp_hours[1])
        if interp_start > ta_end:
            interp_start = api.no_utctime
            interp_end = api.no_utctime
        if interp_end > ta_end:
            interp_end = ta_end

        # Re-organize data before sending to api.quantile_map_forecast. For each source type and geo_point, group
        # forecasts as TsVectorSets, send to api.quantile_map_forecast and return results as ensemble of source-keyed
        # dictionaries of  geo-ts
        # First re-organize weights - one weight per TVS.
        weight_sets = api.DoubleVector([w for ws in weights for w in ws])

        # New version
        results = [{} for i in range(nb_prior_scenarios)]
        for src in input_source_types:
            qm_scenarios = []
            for geo_pt_idx, geo_pt in enumerate(geo_points):
                forecast_sets = api.TsVectorSet()
                for i, fcst_group in enumerate(prep_fcst_lst) :
                   for j, forecast in enumerate(fcst_group):
                        scenarios = api.TsVector()
                        for member in forecast:
                            scenarios.append(member[src][geo_pt_idx].ts)
                        forecast_sets.append(scenarios)
                        if i == self.repo_prior_idx and j==0:
                            prior_data = scenarios
                            # TODO: read prior if repo_prior_idx is None

                qm_scenarios.append(api.quantile_map_forecast(forecast_sets, weight_sets, prior_data, ta,
                                                         interp_start, interp_end, True))

            # Alternative: convert to array to enable slicing
            # arr = np.array(qm_scenarios)

            # Now organize to desired output format: ensemble of source-keyed dictionaries of  geo-ts
            for i in range(0,nb_prior_scenarios):
            # source_dict = {}
                # ts_vct = arr[:, i]
                ts_vct = [x[i] for x in qm_scenarios]
                vct = self.source_vector_map[src]()
                [vct.append(self.source_type_map[src](geo_pt, ts)) for geo_pt, ts in zip(geo_points, ts_vct)]
                # Alternatives:
                # vct[:] = [self.source_type_map[src](geo_pt, ts) for geo_pt, ts in zip(geo_points, ts_vct)]
                # vct = self.source_vector_map[src]([self.source_type_map[src](geo_pt, ts) for geo_pt, ts in zip(geo_points, ts_vct)])
                results[i][src] = vct
        return results

    def get_forecast_ensembles(self):
        # should replace 'get_forecast_ensemble' in the long-run
        pass

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
        ensemble: list of same type as get_timeseries
        Important notice: The returned forecast time-series should at least cover the
            requested period. It could return *more* data than in
            the requested period, but must return sufficient data so
            that the f(t) can be evaluated over the requested period.
        """
        # TODO: generalise handling of geo_criteria by passing it on to prep_fcst_lst
        # TODO: Use utc_period to return scenarios for entire utc_period
        # For time being assume geo_location_criteria is a bounding box
        bbox = geo_location_criteria

        start_time = t_c

        raw_fcst_lst = self._read_fcst(start_time, input_source_types, bbox)

        if self.repo_prior_idx is None:
            prior = self._read_prior()
            nb_prior_scenarios = len(prior)
        else:
            nb_prior_scenarios = len(raw_fcst_lst[self.repo_prior_idx][0])

        # Sort weights based on start time of fcst for first source_type  at first geo point from high to low
        weights = [cl['w'] for cl in self.qm_cfg_params]
        weights = [[w[i] for i in np.argsort([-f[0][input_source_types[0]][0].ts.time(0) for f in f_cl])] for w, f_cl in
                   zip(weights, raw_fcst_lst)]

        results = {}
        for key in self.qm_resolution:
            qm_resolution_idx, start_hour, time_step, nb_time_steps = self.qm_resolution[key]
            ta_start = start_time + api.deltahours(start_hour)
            ta = api.TimeAxis(ta_start, api.deltahours(time_step), nb_time_steps)
            prep_fcst_lst, geo_points = self._prep_fcst(start_time, raw_fcst_lst, qm_resolution_idx, ta)

            # TODO: need to prep prior if self.repo_prior_idx is None
            results[key] = self._call_qm(prep_fcst_lst, weights, geo_points, ta, input_source_types, nb_prior_scenarios)
        return results

    def _call_qm_old(self, prep_fcst_lst, weights, geo_points, ta, input_source_types, nb_prior_scenarios):

        # TODO: Extend handling to cover all cases and send out warnings if interpolation period is modified
        # Check ta against interpolation start and end times
        # Simple logic for time being, should be refined for the overlap cases
        ta_start = ta.time(0)
        ta_end = ta.time(ta.size() - 1)  # start of last time step
        interp_start = ta_start + api.deltahours(self.qm_interp_hours[0])
        interp_end = ta_start + api.deltahours(self.qm_interp_hours[1])
        if interp_start > ta_end:
            interp_start = api.no_utctime
            interp_end = api.no_utctime
        if interp_end > ta_end:
            interp_end = ta_end

        # Re-organize data before sending to api.quantile_map_forecast. For each source type and geo_point, group
        # forecasts as TsVectorSets, send to api.quantile_map_forecast and return results as ensemble of source-keyed
        # dictionaries of  geo-ts
        # First re-organize weights - one weight per TVS.
        weight_sets = api.DoubleVector([w for ws in weights for w in ws])

        # New version
        # results = [{} for i in range(nb_prior_scenarios)]
        # for src in input_source_types:
        #     qm_scenarios = []
        #     for geo_pt_idx, geo_pt in enumerate(geo_points):
        #         forecast_sets = api.TsVectorSet()
        #         for i, fcst_group in enumerate(prep_fcst_lst) :
        #            for j, forecast in enumerate(fcst_group):
        #                 scenarios = api.TsVector()
        #                 for member in forecast:
        #                     scenarios.append(member[src][geo_pt_idx].ts)
        #                 forecast_sets.append(scenarios)
        #                 if i == self.repo_prior_idx and j==0:
        #                     prior_data = scenarios
        #                     # TODO: read prior if repo_prior_idx is None
        #
        #         qm_scenarios.append(api.quantile_map_forecast(forecast_sets, weight_sets, prior_data, ta,
        #                                                  interp_start, interp_end, True))
        #
        #     # Alternative: convert to array to enable slicing
        #     # arr = np.array(qm_scenarios)
        #
        #     # Now organize to desired output format: ensemble of source-keyed dictionaries of  geo-ts
        #     for i in range(0,nb_prior_scenarios):
        #     # source_dict = {}
        #         # ts_vct = arr[:, i]
        #         ts_vct = [x[i] for x in qm_scenarios]
        #         vct = self.source_vector_map[src]()
        #         [vct.append(self.source_type_map[src](geo_pt, ts)) for geo_pt, ts in zip(geo_points, ts_vct)]
        #         # Alternatives:
        #         # vct[:] = [self.source_type_map[src](geo_pt, ts) for geo_pt, ts in zip(geo_points, ts_vct)]
        #         # vct = self.source_vector_map[src]([self.source_type_map[src](geo_pt, ts) for geo_pt, ts in zip(geo_points, ts_vct)])
        #         results[i][src] = vct
        # return results

        # Old version
        weight_sets = api.DoubleVector([w for ws in weights for w in ws])
        dict = {}
        for src in input_source_types:
            qm_scenarios = []
            for geo_pt_idx, geo_pt in enumerate(geo_points):
                forecast_sets = api.TsVectorSet()
                for i, fcst_group in enumerate(prep_fcst_lst):
                    for j, forecast in enumerate(fcst_group):
                        scenarios = api.TsVector()
                        for member in forecast:
                            scenarios.append(member[src][geo_pt_idx].ts)
                        forecast_sets.append(scenarios)

                        # TODO: handle prior similarly if repo_prior_idx is None
                        if i == self.repo_prior_idx and j == 0:
                            prior_data = scenarios

                qm_scenarios.append(api.quantile_map_forecast(forecast_sets, weight_sets, prior_data, ta,
                                                              interp_start, interp_end, True))
            dict[src] = np.array(qm_scenarios)

        # Now organize to desired output format: ensenble of source-keyed dictionaries of  geo-ts
        # TODO: write function to extract info about prior like number of scenarios
        nb_prior_scenarios = dict[input_source_types[0]].shape[1]
        results = []
        for i in range(0, nb_prior_scenarios):
            source_dict = {}
            for src in input_source_types:
                ts_vct = dict[src][:, i]
                vct = self.source_vector_map[src]()
                [vct.append(self.source_type_map[src](geo_pt, ts)) for geo_pt, ts in zip(geo_points, ts_vct)]
                # Alternatives:
                # vct[:] = [self.source_type_map[src](geo_pt, ts) for geo_pt, ts in zip(geo_points, ts_vct)]
                # vct = self.source_vector_map[src]([self.source_type_map[src](geo_pt, ts) for geo_pt, ts in zip(geo_points, ts_vct)])
                source_dict[src] = vct
            results.append(source_dict)
        return results