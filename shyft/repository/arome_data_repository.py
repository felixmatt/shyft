from functools import partial
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from pyproj import transform
from shyft.api import Timeaxis
from shyft.api import TsFactory
from shyft.api import DoubleVector_FromNdArray
from shyft.api import RelHumSource
from shyft.api import TemperatureSource
from shyft.api import PrecipitationSource
from shyft.api import WindSpeedSource
from shyft.api import RadiationSource
from shyft.api import GeoPoint
from shyft.api import UtcPeriod


class AromeDataRepositoryException(Exception):
    pass


class AromeDataRepository(object):

    def __init__(self, filename, epsg_id, bounding_box, utc_period,
                 x_padding=5000.0, y_padding=5000.0, fields=None):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP model,
        and initialize data retrieval.

        Parameters
        ----------
        filename: string
            Name of netcdf file containing spatially distributed input fata
        epsg_id: int
            Unique coordinate system id for result coordinates. Currently 32632
            and 32633 are supperted.
        bounding_box: list
            A list on the form [[x_ul, x_ur, x_lr, x_ll],
            [y_ul, y_ur, y_lr, y_ll]] describing the outer boundaries of the
            domain that shoud be extracted. Coordinates are given in epgs_id
            coordinate system.
        utc_period: api.UtcPeriod
            period to fetch such that utc_period.start and utc_period.end are 
            both included in the interval, if possible
        x_padding: float
            Longidutinal padding in meters, added both east and west
        y_padding: float
            Latitudinal padding in meters, added both north and south
        fields: list
            List of data field names to extract: ["relative_humidity",
            "temperature", "z", "precipitation", "x_wind", "y_wind",
            "radiation"]. If not given, all fields are extracted.


        Arome NWP model output is from:
        Catalog http://thredds.met.no/thredds/catalog/arome25/catalog.html

        Contact:
            Name: met.no
            Organization: met.no
            Email: thredds@met.no
            Phone: +47 22 96 30 00
        """
        self.shyft_cs = "+proj=utm +zone={} +ellps={} +datum={} \
 +units=m +no_defs".format(epsg_id - 32600, "WGS84", "WGS84")
        dataset = Dataset(filename)
        data_vars = dataset.variables

        if isinstance(utc_period, UtcPeriod):
            utc_period = [utc_period.start, utc_period.end]

        # Extract time dimension
        time = data_vars["time"][:]
        dts = time[1:] - time[:-1]

        idx_min = time.searchsorted(utc_period[0], side='left')
        idx_max = time.searchsorted(utc_period[1], side='right')
        time_slice = slice(idx_min, idx_max)
        extract_subset = True if time[time_slice].shape != time.shape else False
        time = time[time_slice]

        # Add a padding to the bounding box to make sure the computational
        # domain is fully enclosed in arome dataset
        bounding_box = np.array(bounding_box)
        bounding_box[0][0] -= x_padding
        bounding_box[0][1] += x_padding
        bounding_box[0][2] += x_padding
        bounding_box[0][3] -= x_padding
        bounding_box[1][0] += x_padding
        bounding_box[1][1] += x_padding
        bounding_box[1][2] -= x_padding
        bounding_box[1][3] -= x_padding

        # Field names and mappings
        netcdf_data_fields = ["relative_humidity_2m",
                              "air_temperature_2m",
                              "altitude",
                              "precipitation_amount",
                              "x_wind_10m",
                              "y_wind_10m",
                              "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time"]

        # arome data and time conversions, ordered as netcdf_data_fields
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
                return Timeaxis(int(t[0]), int(t[1] - t[0]), len(t) - shift)

            def noop(d):
                return d[time_slice]
            
            def prec(d):
                return d[1:][time_slice]
            t_to_ta_0 = partial(t_to_ta, t, 0)  # Full
            t_to_ta_1 = partial(t_to_ta, t, 1)
            return [(noop, t_to_ta_0),
                    (lambda air_temp: air_temp[time_slice] - 273.15, t_to_ta_0),
                    (lambda x: x, lambda: None), # Altitude
                    (prec, t_to_ta_1),
                    (noop, t_to_ta_0),
                    (noop, t_to_ta_0),
                    (lambda rad: np.clip(((rad[1:][time_slice] - rad[:-1][time_slice])/(dts[time_slice, np.newaxis, np.newaxis,
                                          np.newaxis])), 0.0, 1000.0),
                     t_to_ta_1)]

        shyft_data_fields = ["relative_humidity",
                             "temperature",
                             "z",
                             "precipitation",
                             "x_wind",
                             "y_wind",
                             "radiation"]

        self.source_type_map = {"relative_humidity": RelHumSource,
                                "temperature": TemperatureSource,
                                "precipitation": PrecipitationSource,
                                "radiation": RadiationSource,
                                "wind_speed": WindSpeedSource}

        if fields is None:
            fields = shyft_data_fields
        else:
            assert all([field in shyft_data_fields for field in fields])

        shyft_net_map = {s: n for n, s in zip(netcdf_data_fields,
                                              shyft_data_fields)}
        data_convert_map = {s: c for s, c in zip(shyft_data_fields,
                                                 netcdf_data_convert(time))}

        # Target projection
        shyft_proj = Proj(self.shyft_cs)

        raw_data = {}
        data_proj = None
        self.xx = self.yy = self.extracted_data = None
        for data_field in fields:
            if not shyft_net_map[data_field] in data_vars.keys():
                continue
            data = data_vars[shyft_net_map[data_field]]

            if data_proj is None:
                # Get coordinate system for arome data
                data_cs = str(data_vars[data.grid_mapping].proj4)
                data_cs += " +towgs84=0,0,0"
                data_proj = Proj(data_cs)

                # Find bounding box in arome projection
                bb_proj = transform(shyft_proj, data_proj,
                                    bounding_box[0], bounding_box[1])
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
                self.xx, self.yy = transform(data_proj, shyft_proj,
                                             *np.meshgrid(x[x_inds],
                                                          y[y_inds]))

            # Construct slice
            data_slice = len(data.dimensions)*[slice(None)]
            data_slice[data.dimensions.index("x")] = x_inds
            data_slice[data.dimensions.index("y")] = y_inds

            # Add extracted data and corresponding coordinates to class
            raw_data[data_field] = data[data_slice]
        extracted_data = {key: (data_convert_map[key][0](raw_data[key]),
                                data_convert_map[key][1]())
                          for key in raw_data}
        if "x_wind" in extracted_data.keys() and \
                "y_wind" in extracted_data.keys():
            x_wind, _ = extracted_data.pop("x_wind")
            y_wind, t = extracted_data.pop("y_wind")
            extracted_data["wind_speed"] = np.sqrt(np.square(x_wind) +
                                                   np.square(y_wind)), t
        self.time_series, self.other_data = \
            self._convert_to_time_series(extracted_data)

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

    def _convert_to_time_series(self, extracted_data):
        """Convert data from numpy structures to shyft data.

        We assume the time axis is regular, and that we can use a point time
        series with a parametrized time axis definition and corresponding
        vector of values.
        """
        time_series = {}
        non_time_series = {}
        tsc = TsFactory().create_point_ts
        for key, (data, ta) in extracted_data.iteritems():
            if ta is None:
                non_time_series[key] = data
                continue
            fslice = (len(data.shape) - 2)*(slice(None),)
            I, J = data.shape[-2:]

            def construct(d):
                return tsc(ta.size(), ta.start(), ta.delta(),
                           DoubleVector_FromNdArray(d.flatten()), 0)
            time_series[key] = np.array([[construct(data[fslice + (i, j)])
                                         for j in xrange(J)] for i in
                                         xrange(I)])
        return time_series, non_time_series

    def add_time_series(self, other, eps=1.0e-10):
        """Add other's timeseries to self

        Add all the time series from the other repository to this, if the x,y
        locations match within a tolerance.

        Parameters
        ----------
        other: AromeDataRepository
            Repository with additional time series and lat/long coordinates
        eps: float, optional
            Tolerance for point co-location
        """
        if np.linalg.norm(other.xx.ravel() - self.xx.ravel()) > eps or \
           np.linalg.norm(other.yy.ravel() - self.yy.ravel()) > eps:
            raise AromeDataRepositoryException()
        self.time_series.update(other.time_series)

    def fetch_sources(self, keys=None):
        """Get shyft source vectors for keys

        Convert timeseries and geo locations, to corresponding input sources,
        and return these as shyft source vectors.

        Parameters
        ----------
        keys: list, optional
            If given, a list of data names to extract input source vectors for.
        """
        if "geo_points" not in self.other_data:
            self.other_data["geo_points"] = self._geo_points()
        pts = self.other_data["geo_points"]
        if keys is None:
            keys = self.time_series.keys()
        elif isinstance(keys, str):
            keys = [keys]
        sources = {}
        all_ = slice(None)
        for key in keys:
            ts = self.time_series[key]
            if key not in self.source_type_map:
                continue
            tpe = self.source_type_map[key]
            sources[key] = tpe.vector_t([tpe(GeoPoint(*pts[idx + (all_,)]),
                                         ts[idx]) for idx in
                                         np.ndindex(pts.shape[:-1])])
        return sources
