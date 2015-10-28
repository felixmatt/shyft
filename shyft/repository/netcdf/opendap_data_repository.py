from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from pyproj import transform
from shyft import api
from .. import interfaces


class GFSDataRepositoryError(Exception):
    pass


class GFSDataRepository(interfaces.GeoTsRepository):
    """
    Repository for geo located time series given as GFS data in an OpenDAP
    server.

    Hydrologic data:
    $ hyd_ds = Dataset("http://nomads.ncep.noaa.gov:9090/dods/gens/gens20151015/gec00_00z")
    Shoud work, but does not:
    alt_ds = Dataset("http://cwcgom.aoml.noaa.gov/erddap/griddap/etopo360")

    ETOPO1 Spatial Reference System:
    urn:ogc:def:crs:EPSG::4326 urn:ogc:def:crs:EPSG::5715
    I interpret this as EPSG::4326 on land, and EPSG::5715 for sea depths.


    # This is ok:
    res = urllib2.urlopen("http://cwcgom.aoml.noaa.gov/erddap/griddap/etopo180.nc?altitude[(57):1:(71.5)][(3):1:(32)]")
    tf = open("etopo180.nc", "wb")
    tf.write(res.read())
    tf.close()
    alt_ds2 = Dataset("etopo180.nc")

    For data specification, see: http://nomads.ncdc.noaa.gov/thredds

    Specifically, we assume the following the the data set:
        * Root group has variables:
            * tmp2m: float32 array of dims (ens, time, lat, long) and units K (temperature)
            * pratesfc: float32 array of dims (ens, time, lat, long) and units kg m-2 (precip)
            *
    """

    # Constants to convert from days since 1.1.1:0:0:0 to secs since epoch
    __time_a = 3600*24.0  # utc = (t_gfs - self.time_b)*self.time_a
    __time_b = 719164.0

    def __init__(self, epsg, dem_file, utc=None, bounding_box=None):
        self.epsg = epsg
        self.shyft_cs = "+init=EPSG:32632"
        self.dem_file = dem_file
        self.ensemble_idx = 0
        self.base_url = "http://nomads.ncep.noaa.gov:9090/dods/gens"
        if utc is not None:
            ens = 0  # Choose zero ensemble by default
            cal = api.Calendar()
            ymd = cal.calendar_units(utc)
            self.gfs_url = "{}/gens{:04d}{:02d}{:02d}/gec{:02d}_{:02d}z".format(self.base_url,
                                                                                ymd.year,
                                                                                ymd.month,
                                                                                ymd.day,
                                                                                ens,
                                                                                ymd.hour//6*6)
        else:
            self.grf_url = None
        self.bounding_box = bounding_box

        self._gfs_shyft_map = {"ugrd10m": "x_wind",
                               "vgrd10m": "y_wind",
                               "tmp2m": "temperature",
                               "pratesfc": "precipitation",
                               "rh2m": "relative_humidity",
                               "dswrfsfc": "radiation"}
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
        if self.gfs_url is None:
            raise GFSDataRepositoryError("Repository not initialized properly "
                                         "to call get_timeseries directly.")

        with Dataset(self.gfs_url) as dataset:
            return self._get_ensemble_data_from_dataset(dataset, input_source_types,
                                                        utc_period, geo_location_criteria)

    def _get_ensemble_data_from_dataset(self, dataset, input_source_types,
                                        utc_period, geo_location_criteria): 
        if geo_location_criteria is not None:
            self.bounding_box = geo_location_criteria

        if "wind_speed" in input_source_types:
            input_source_types = list(input_source_types)  # Copy the possible mutable input list
            input_source_types.remove("wind_speed")
            input_source_types.extend(["x_wind", "y_wind"])

        raw_data = {}
        lon = dataset.variables.get("lon", None)
        lat = dataset.variables.get("lat", None)
        time = dataset.variables.get("time", None)
        if not all([lon, lat, time]):
            raise GFSDataRepositoryError("Something is wrong with the dataset."
                                         " lat/lon coords or time not found.")
        time = self.ad_to_utc(time)  # Fetch all times
        idx_min = time.searchsorted(utc_period.start, side='left')
        idx_max = time.searchsorted(utc_period.end, side='right')
        time_slice = slice(idx_min, idx_max)
        time = time[time_slice]

        x, y, _, (m_lon, m_lat), _ = self._limit(lon[:], lat[:], self.shyft_cs)

        for k in dataset.variables.keys():
            if self._gfs_shyft_map.get(k, None) in input_source_types:
                data = dataset.variables[k]
                data_slice = len(data.dimensions)*[slice(None)]
                data_slice[data.dimensions.index("ens")] = self.ensemble_idx
                data_slice[data.dimensions.index("lon")] = m_lon
                data_slice[data.dimensions.index("lat")] = m_lat
                data_slice[data.dimensions.index("time")] = time_slice
                raw_data[self._gfs_shyft_map[k]] = data[data_slice]
        with Dataset(self.dem_file) as dataset:
            alts = dataset.variables["altitude"]
            lats = dataset.variables["latitude"][:]
            longs = dataset.variables["longitude"][:]
            alts = alts[np.round(lats) == lats, np.round(longs) == longs]
            lats = lats[np.round(lats) == lats]
            longs = longs[np.round(longs) == longs]
            _x, _y, z, (_m_lon, _m_lat), _ = self._limit(longs, lats, self.shyft_cs, alts)
            assert np.linalg.norm(x - _x + y - _y) < 1.0e-10  # x/y coordinates must match
            pts = np.dstack((x, y, z)).reshape(*(x.shape + (3,)))
        if set(("x_wind", "y_wind")).issubset(raw_data):
            x_wind = raw_data.pop("x_wind")
            y_wind = raw_data.pop("y_wind")
            raw_data["wind_speed"] = np.sqrt(np.square(x_wind) + np.square(y_wind))
        extracted_data = self._transform_raw(raw_data, time)
        return self._geo_ts_to_vec(self._convert_to_timeseries(extracted_data), pts)

    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria=None):
        """See base class."""
        ens = 0  # Choose zero ensemble by default
        cal = api.Calendar()
        ymd = cal.calendar_units(t_c)
        self.gfs_url = "{}/gens{:04d}{:02d}{:02d}/gec{:02d}_{:02d}z".format(self.base_url,
                                                                            ymd.year,
                                                                            ymd.month,
                                                                            ymd.day,
                                                                            ens,
                                                                            ymd.hour//6*6)
        return self.get_timeseries(input_source_types, utc_period, geo_location_criteria)

    def get_forecast_ensemble(self, input_source_types, utc_period,
                              t_c, geo_location_criteria=None):
        """See base class: ..interfaces.GeoTsRepository"""
        cal = api.Calendar()
        ymd = cal.calendar_units(t_c)
        res = []
        gfs_url = ("{}/gens{:04d}{:02d}"
                   "{:02d}/gep_all_{:02d}z".format(self.base_url,
                                                   ymd.year,
                                                   ymd.month,
                                                   ymd.day,
                                                   ymd.hour//6*6))
        with Dataset(gfs_url) as dataset:
            for ens in range(21):
                self.ensemble_idx = ens
                res.append(self._get_ensemble_data_from_dataset(dataset, input_source_types, utc_period, geo_location_criteria))
        return res

        

    def _transform_raw(self, data, time):

        def noop_time(t):
            return api.Timeaxis(t[0], t[1] - t[0], len(t))

        def noop_space(x):
            return x

        def air_temp_conv(x):
            return x - 273.15

        def prec_conv(x):
            return x*3600

        convert_map = {"wind_speed": lambda x, t: (noop_space(x), noop_time(t)),
                       "radiation": lambda x, t: (noop_space(x), noop_time(t)),
                       "temperature": lambda x, t: (air_temp_conv(x), noop_time(t)),
                       "precipitation": lambda x, t: (prec_conv(x), noop_time(t)),
                       "relative_humidity": lambda x, t: (noop_space(x), noop_time(t))}
        res = {}
        for k, v in data.items():
            res[k] = convert_map[k](v, time)
        return res

    def _convert_to_timeseries(self, data):
        tsc = api.TsFactory().create_point_ts
        time_series = {}
        for key, (data, ta) in data.items():
            fslice = (len(data.shape) - 2)*[slice(None)]
            I, J = data.shape[-2:]

            def construct(d):
                if ta.size() != d.size:
                    raise GFSDataRepositoryError("Time axis size {} not equal to the number of "
                                                   "data points ({}) for {}"
                                                   "".format(ta.size(), d.size, key))
                return tsc(ta.size(), ta.start(), ta.delta(),
                           api.DoubleVector_FromNdArray(d.flatten()), 0)

            time_series[key] = np.array([[construct(data[fslice + [i, j]])
                                          for j in range(J)] for i in range(I)])
        return time_series

    def _geo_ts_to_vec(self, data, pts):
        res = {}
        for name, ts in data.items():
            tpe = self.source_type_map[name] 
            ids = [idx for idx in np.ndindex(pts.shape[:-1])]
            res[name] = tpe.vector_t([tpe(api.GeoPoint(*pts[idx]),
                                      ts[idx]) for idx in np.ndindex(pts.shape[:-1])])
        return res

    def _limit(self, lon, lat, target_cs, altitudes=None):
        data_proj = Proj("+init=EPSG:4326")  # WGS84, TODO: How to treat vertical transform?
        target_proj = Proj(target_cs)

        # Find bounding box in arome projection
        bbox = self.bounding_box
        bb_proj = transform(target_proj, data_proj, bbox[0], bbox[1])
        lon_min, lon_max = min(bb_proj[0]), max(bb_proj[0])
        lat_min, lat_max = min(bb_proj[1]), max(bb_proj[1])

        # Limit data
        lon_upper = lon >= lon_min
        lon_lower = lon <= lon_max
        if sum(lon_upper == lon_lower) < 2:
            lon_upper[np.argmax(lon_upper) - 1] = True
            lon_lower[np.argmin(lon_lower)] = True

        lat_upper = lat >= lat_min
        lat_lower = lat <= lat_max
        if sum(lat_upper == lat_lower) < 2:
            lat_upper[np.argmax(lat_upper) - 1] = True
            lat_lower[np.argmin(lat_lower)] = True
        lon_inds = np.nonzero(lon_upper == lon_lower)[0]
        lat_inds = np.nonzero(lat_upper == lat_lower)[0]
        # Masks
        lon_mask = lon_upper == lon_lower
        lat_mask = lat_upper == lat_lower

        if lon_inds.size == 0:
            raise GFSDataRepositoryError("Bounding box longitudes don't intersect with dataset.")
        if lat_inds.size == 0:
            raise GFSDataRepositoryError("Bounding box latitudes don't intersect with dataset.")

        if altitudes is not None:
            alts = np.clip(altitudes[lat_inds[:, None], lon_inds], 0.0, 100000)
            xyz = np.meshgrid(lon[lon_inds], lat[lat_inds]) + [alts]
            t_xyz = transform(data_proj, target_proj, xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel())
            x, y, z = [tmp.reshape(xyz[0].shape) for tmp in t_xyz]
        else:
            x, y = transform(data_proj, target_proj, *np.meshgrid(lon[lon_inds], lat[lat_inds]))
            z = None
        return x, y, z, (lon_mask, lat_mask), (lon_inds, lat_inds)

    @classmethod
    def ad_to_utc(cls, T):
        return np.array((np.asarray(T) - cls.__time_b)*cls.__time_a, dtype='int')


if __name__ == "__main__":
    utcs = GFSDataRepository.ad_to_utc([719164, 735887.0])
    cal = api.Calendar()
    print([cal.to_string(utc) for utc in utcs])
