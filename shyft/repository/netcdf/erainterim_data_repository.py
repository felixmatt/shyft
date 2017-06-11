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
from shyft import shyftdata_dir
from .. import interfaces
from .time_conversion import convert_netcdf_time
#from repository import interfaces
#from repository.netcdf.time_conversion import convert_netcdf_time


class ERAInterimDataRepositoryError(Exception):
    pass


class ERAInterimDataRepository(interfaces.GeoTsRepository):
    """
    Repository for geo located timeseries stored in netCDF files.

    """
    
    # Constants used in RH calculation
    __a1_w=611.21 # Pa
    __a3_w=17.502
    __a4_w=32.198 # K

    __a1_i=611.21 # Pa
    __a3_i=22.587
    __a4_i=-20.7 # K

    __T0=273.16 # K
    __Tice=205.16 # K
                     
    #def __init__(self, params, region_config):
    def __init__(self, epsg, filename, bounding_box=None):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP model,
        and initialize data retrieval.
        """
        #self._rconf = region_config
        #epsg = self._rconf.domain()["EPSG"]
        #filename = params["stations_met"]

        #if not path.isdir(directory):
        #    raise CFDataRepositoryError("No such directory '{}'".format(directory))
        filename = path.expandvars(filename)
        if not path.isabs(filename):
            # Relative paths will be prepended the data_dir
            filename = path.join(shyftdata_dir, filename)
        if not path.isfile(filename):
            raise ERAInterimDataRepositoryError("No such file '{}'".format(filename))
            
        self._filename = filename # path.join(directory, filename)
        self.allow_subset = True # allow_subset
        
        #self.elevation_file = None
        self.analysis_hours = [0,12]
        self.cal = api.Calendar()

        self.shyft_cs = "+init=EPSG:{}".format(epsg)
        #self._bounding_box = None # bounding_box
        self.bounding_box = bounding_box

        # Field names and mappings netcdf_name: shyft_name
        self._era_shyft_map = {"u10": "x_wind",
                               "v10": "y_wind",
                               "t2m": "temperature",
                               "tp": "precipitation",
                               "sp": "surface_pressure",
                               "d2m": "dewpoint_temperature",
                               "ssrd": "radiation"}

        self._shift_fields = ("tp","ssrd")

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
            raise ERAInterimDataRepositoryError("File '{}' not found".format(filename))
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

    def _convert_to_timeseries(self, data):
        tsc = api.TsFactory().create_point_ts
        time_series = {}
        for key, (data, ta) in data.items():
            fslice = (len(data.shape) - 2)*[slice(None)]
            I, J = data.shape[-2:]

            def construct(d):
                if ta.size() != d.size:
                    raise ERAInterimDataRepositoryError("Time axis size {} not equal to the number of "
                                                   "data points ({}) for {}"
                                                   "".format(ta.size(), d.size, key))
                return tsc(ta.size(), ta.start, ta.delta_t,
                           api.DoubleVector_FromNdArray(d.flatten()), self.series_type[key])

            time_series[key] = np.array([[construct(data[fslice + [i, j]])
                                          for j in range(J)] for i in range(I)])
        return time_series

    def _limit(self, lon, lat, target_cs): # TODO: lat long boundaries are not rectangular
        data_proj = Proj("+init=EPSG:4326")  # WGS84
        target_proj = Proj(target_cs)

        # Find bounding box in ERA projection
        bbox = self.bounding_box
        bb_proj = transform(target_proj, data_proj, bbox[0], bbox[1])
        lon_min, lon_max = min(bb_proj[0]), max(bb_proj[0])
        lat_min, lat_max = min(bb_proj[1]), max(bb_proj[1])
        #print(lon_min,lon_max,lat_min,lat_max)

        # Limit data
        lon_upper = lon >= lon_min
        lon_lower = lon <= lon_max

        lat_upper = lat >= lat_min
        lat_lower = lat <= lat_max

        lon_inds = np.nonzero(lon_upper == lon_lower)[0]
        lat_inds = np.nonzero(lat_upper == lat_lower)[0]
        # Masks
        lon_mask = lon_upper == lon_lower
        lat_mask = lat_upper == lat_lower
        
        #print (lon_inds,lat_inds)
        #print (lon[lon_inds],lat[lat_inds])

        if lon_inds.size == 0:
            raise ERAInterimDataRepositoryError("Bounding box longitudes don't intersect with dataset.")
        if lat_inds.size == 0:
            raise ERAInterimDataRepositoryError("Bounding box latitudes don't intersect with dataset.")

        x, y = transform(data_proj, target_proj, *np.meshgrid(lon[lon_inds], lat[lat_inds]))

        return x, y, (lon_mask, lat_mask), (lon_inds, lat_inds)

    def _get_data_from_dataset(self, dataset, input_source_types, utc_period,
                               geo_location_criteria, ensemble_member=None):

        if geo_location_criteria is not None:
            self.bounding_box = geo_location_criteria

        if "wind_speed" in input_source_types:
            input_source_types = list(input_source_types)  # Copy the possible mutable input list
            input_source_types.remove("wind_speed")
            input_source_types.extend(["x_wind", "y_wind"])
        no_temp = False
        if "temperature" not in input_source_types: no_temp = True
        if "relative_humidity" in input_source_types:
            input_source_types.remove("relative_humidity")
            input_source_types.extend(["surface_pressure", "dewpoint_temperature"])
            if no_temp: input_source_types.extend(["temperature"])

        raw_data = {}
        lon = dataset.variables.get("longitude", None)
        lat = dataset.variables.get("latitude", None)
        time = dataset.variables.get("time", None)

        if not all([lon, lat, time]):
            raise ERAInterimDataRepositoryError("Something is wrong with the dataset."
                                         " lat/lon coords or time not found.")
        time = convert_netcdf_time(time.units,time)
        #print (time[0])
        #t_indx = np.argsort(time)
        #time = time[t_indx]
        #self.time=time
        
        idx_min = time.searchsorted(utc_period.start, side='left')
        idx_max = time.searchsorted(utc_period.end, side='right')
        issubset = True if idx_max < len(time) - 1 else False
        time_slice = slice(idx_min, idx_max)  
        #print (idx_min, idx_max)

        x, y, (m_lon, m_lat), _ = self._limit(lon[:], lat[:], self.shyft_cs)

        for k in dataset.variables.keys():
            if self._era_shyft_map.get(k, None) in input_source_types:
                if k in self._shift_fields and issubset:  # Add one to time slice
                    data_time_slice = slice(time_slice.start, time_slice.stop + 1)
                else:
                    data_time_slice = time_slice
                data = dataset.variables[k]
                data_slice = len(data.dimensions)*[slice(None)]
                #data_slice[data.dimensions.index("ens")] = self.ensemble_idx
                data_slice[data.dimensions.index("longitude")] = m_lon
                data_slice[data.dimensions.index("latitude")] = m_lat
                data_slice[data.dimensions.index("time")] = data_time_slice

                pure_arr = data[data_slice]

                if isinstance(pure_arr, np.ma.core.MaskedArray):
                    #print(pure_arr.fill_value)
                    pure_arr = pure_arr.filled(np.nan)
                raw_data[self._era_shyft_map[k]] = pure_arr, k
                
        if "z" in dataset.variables.keys():
            data = dataset.variables["z"]
            dims = data.dimensions
            data_slice = len(data.dimensions)*[slice(None)]
            data_slice[dims.index("longitude")] = m_lon
            data_slice[dims.index("latitude")] = m_lat
            z = data[data_slice]/9.80665 # Converting from geopotential to m
        else:
            raise ERAInterimDataRepositoryError("No elevations found in dataset")
        pts = np.dstack((x, y, z)).reshape(*(x.shape + (3,)))
        
        # Make sure requested fields are valid, and that dataset contains the requested data.
        if not self.allow_subset and not (set(raw_data.keys()).issuperset(input_source_types)):
            raise ERAInterimDataRepositoryError("Could not find all data fields")
            
        if set(("x_wind", "y_wind")).issubset(raw_data):
            x_wind, _ = raw_data.pop("x_wind")
            y_wind, _ = raw_data.pop("y_wind")
            raw_data["wind_speed"] = np.sqrt(np.square(x_wind) + np.square(y_wind)), "wind_speed"
        if set(("surface_pressure", "dewpoint_temperature")).issubset(raw_data):
            sfc_p, _ = raw_data.pop("surface_pressure")
            dpt_t, _ = raw_data.pop("dewpoint_temperature")
            if no_temp:
                sfc_t, _ = raw_data.pop("temperature")
            else:
                sfc_t, _ = raw_data["temperature"]
            raw_data["relative_humidity"] = self.calc_RH(sfc_t,dpt_t,sfc_p), "relative_humidity"
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
            return T - 273.15

        def prec_acc_conv(p):
            indx = np.nonzero([self.cal.calendar_units(int(ti)).hour in self.analysis_hours for ti in time])[0]
            f = 1000.*api.deltahours(1)/(time[1] - time[0]) # conversion from m/delta_t to mm/1hour
            dp = np.clip((p[1:] - p[:-1])*f, 0.0, 10000.) # np.clip b/c negative values may occur after deaccumulation
            dp[indx] = p[indx+1]*f
            return dp

        def rad_conv(r):
            indx = np.nonzero([self.cal.calendar_units(int(ti)).hour in self.analysis_hours for ti in time])[0]
            dr = np.clip((r[1:] - r[:-1])/(time[1] - time[0]), 0.0, 10000.) # np.clip b/c negative values may occur after deaccumulation
            dr[indx] = r[indx+1]/(time[1] - time[0])
            return dr

        convert_map = {"wind_speed": lambda x, t: (noop_space(x), noop_time(t)),
                       "relative_humidity": lambda x, t: (noop_space(x), noop_time(t)),
                       "t2m": lambda x, t: (air_temp_conv(x), noop_time(t)),
                       "ssrd": lambda x, t: (rad_conv(x), dacc_time(t)),
                       "tp": lambda x, t: (prec_acc_conv(x), dacc_time(t))}
        res = {}
        for k, (v, ak) in data.items():
            res[k] = convert_map[ak](v, time)
        return res

    def _geo_ts_to_vec(self, data, pts):
        res = {}
        for name, ts in iteritems(data):
            tpe = self.source_type_map[name]
            # YSA: It seems that the boost-based interface does not handle conversion straight from list of non-primitive objects
            #res[name] = tpe.vector_t([tpe(api.GeoPoint(*pts[idx]),
            #                          ts[idx]) for idx in np.ndindex(pts.shape[:-1])])
            tpe_v=tpe.vector_t()
            for idx in np.ndindex(pts.shape[:-1]):
                tpe_v.append(tpe(api.GeoPoint(*pts[idx]), ts[idx]))
            res[name] = tpe_v
        return res
        
    
    @classmethod    
    def calc_q(cls,T,p,alpha):
        e_w = cls.__a1_w*np.exp(cls.__a3_w*((T-cls.__T0)/(T-cls.__a4_w)))
        e_i = cls.__a1_i*np.exp(cls.__a3_i*((T-cls.__T0)/(T-cls.__a4_i)))
        q_w = 0.622*e_w/(p-(1-0.622)*e_w)
        q_i = 0.622*e_i/(p-(1-0.622)*e_i)
        return alpha*q_w+(1-alpha)*q_i
        
    @classmethod
    def calc_alpha(cls,T):
        alpha=np.zeros(T.shape,dtype='float')
        #alpha[T<=Tice]=0.
        alpha[T>=cls.__T0]=1.
        indx=(T<cls.__T0)&(T>cls.__Tice)
        alpha[indx]=np.square((T[indx]-cls.__Tice)/(cls.__T0-cls.__Tice))
        return alpha
        
    @classmethod    
    def calc_RH(cls,T,Td,p):
        alpha = cls.calc_alpha(T)
        qsat = cls.calc_q(T,p,alpha)
        q = cls.calc_q(Td,p,alpha)
        return q/qsat
