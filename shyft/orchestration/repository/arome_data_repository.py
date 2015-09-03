from netCDF4 import Dataset
from netCDF4 import num2date
from netCDF4 import timedelta
from pyproj import Proj
from pyproj import transform
from numpy import array
from numpy import intersect1d
from numpy import where
from numpy import meshgrid
from numpy import newaxis
from numpy import zeros
from numpy import clip
from numpy import empty
from numpy import ndindex
from numpy import square
from numpy import sqrt
from numpy.linalg import norm
from matplotlib import pylab as plt
from functools import partial

from shyft.api import Timeaxis
from shyft.api import TsFactory
from shyft.api import DoubleVector_FromNdArray
from shyft.api import RelHumSource
from shyft.api import TemperatureSource
from shyft.api import PrecipitationSource
from shyft.api import WindSpeedSource
from shyft.api import RadiationSource
from shyft.api import GeoPoint


class AromeDataRepositoryException(Exception):
    pass

class AromeDataRepository(object):

    def __init__(self, filename, epsg_id, bounding_box, x_padding=5000.0, y_padding=5000.0, fields=None):
        """
        Construct the netCDF4 dataset reader for data from Arome NWP model, and
        initialize data retrieval.
       
        Parameters
        ----------
        filename: string
            Name of netcdf file containing spatially distributed input fata
        epsg_id: int 
            Unique coordinate system id for result coordinates. Currently 32632 and 32633 are supperted.
        bounding_box: list
            A list on the form [[x_ul, x_ur, x_lr, x_ll], [y_ul, y_ur, y_lr, y_ll]] describing the
            outer boundaries of the domain that shoud be extracted. Coordinates are given in 
            epgs_id coordinate system.
        x_padding: float
            Longidutinal padding in meters, added both east and west
        y_padding: float
            Latitudinal padding in meters, added both north and south
        fields: list
            List of data field names to extract: ["relative_humidity", "temperature", "z",
            "precipitation", "x_wind", "y_wind", "radiation"]. If not given, all fields are extracted.


        Arome NWP model output is from:
        Catalog http://thredds.met.no/thredds/catalog/arome25/catalog.html

        Contact:
            Name: met.no
            Organization: met.no
            Email: thredds@met.no
            Phone: +47 22 96 30 00
        """
        self.shyft_cs = "+proj=utm +zone={} +ellps={} +datum={} +units=m +no_defs".format(epsg_id - 32600,
                                                                                          "WGS84", "WGS84")
        dataset = Dataset(filename)
        self.ds = dataset
        data_vars = dataset.variables

        # Extract time dimension
        time = data_vars["time"]
        t0 = int(time[0])
        data_dt = int(time[1] - time[0])
        nt = len(time)

        # Add a padding to the bounding box to make sure the computational domain is 
        # fully enclosed in arome dataset
        bounding_box = array(bounding_box)
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
            convert from arome data to shyft data. We skip the first values in
            all time series, except from for radiation, where the calculated
            radiation is given by:
            rad[t_i] = sw_flux(t_{i+1}) - sw_flux(t_i)/dt for i in 0, ..., N - 1 
            where N is the number of values in the dataset, and equals the
            number of forcast time points + 1. Also temperatures are converted from Kelvin to Celcius,
            and the elevation data set is treated as a special case.
            """
            t_to_ta = lambda t, shift: Timeaxis(int(t[1 - shift]), int(t[1] - t[0]), len(t) - 1)
            t_to_ta_normal = partial(t_to_ta, t, 0)  # Full 
            t_to_ta_rad = partial(t_to_ta, t, 1)
            noop = lambda d: d[1:]  # First point (at t[0]) is not a forecast, so we skip it
            hnoop = lambda d: d # No time for heights
            return [(noop, t_to_ta_normal),
                    (lambda air_temp: (air_temp - 273.15)[1:], t_to_ta_normal),
                    (hnoop, lambda: None),
                    (noop, t_to_ta_normal),
                    (noop, t_to_ta_normal),
                    (noop, t_to_ta_normal),
                    (lambda rad: clip(((rad[1:] - rad[:-1])/((t[1:] - t[:-1])[:, newaxis, newaxis, 
                                                                             newaxis])), 0.0, 1000.0),
                                                                             t_to_ta_rad)]

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

        net_shyft_map = {n: s for n,s in zip(netcdf_data_fields, shyft_data_fields)}
        shyft_net_map = {s: n for n,s in zip(netcdf_data_fields, shyft_data_fields)}
        data_convert_map = {s: c for s,c in zip(shyft_data_fields, netcdf_data_convert(time))}
        
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
                bb_proj = transform(shyft_proj, data_proj, bounding_box[0], bounding_box[1])
                x_min, x_max = min(bb_proj[0]), max(bb_proj[0])
                y_min, y_max = min(bb_proj[1]), max(bb_proj[1])

                #print x_min, x_max
                #print y_min, y_max

                # Limit data
                x = data_vars["x"][:] 
                x1 = where(x >= x_min)[0]
                x2 = where(x <= x_max)[0]
                x_inds = intersect1d(x1, x2, assume_unique=True)

                y = data_vars["y"][:] 
                y1 = where(y >= y_min)[0]
                y2 = where(y <= y_max)[0]
                y_inds = intersect1d(y1, y2, assume_unique=True)

                # Transform from arome coordinates to shyft coordinates
                self._ox, self._oy = meshgrid(x[x_inds], y[y_inds])
                self.xx, self.yy = transform(data_proj, shyft_proj, *meshgrid(x[x_inds], y[y_inds]))

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
            extracted_data["wind_speed"] = sqrt(square(x_wind) + square(y_wind)), t
            
        self.time_series, self.other_data = self._convert_to_time_series(extracted_data)

    def _construct_geo_points(self):
        """
        Construct a numpy array over the indices with (x,y,z) coordinates at each (i,j).
        """
        pts = empty(self.xx.shape + (3,), dtype='d')
        pts[:, :, 0] = self.xx
        pts[:, :, 1] = self.yy
        pts[:, :, 2] = self.other_data["z"] if "z" in self.other_data else zeros(self.xx.shape, dtype='d')
        return pts

    def _convert_to_time_series(self, extracted_data):
        """
        Convert data from numpy structures to shyft data. We assume the time axis is regular, and 
        that we can use a point time series with a parametrized time axis definition and 
        corresponding vector of values.
        """
        time_series = {}
        non_time_series = {}
        tsf = TsFactory()
        for key, (data, ta) in extracted_data.iteritems():
            if ta is None:
                non_time_series[key] = data
                continue
            fslice = (len(data.shape) - 2)*(slice(None),)
            I, J = data.shape[-2:]
            construct = lambda d: tsf.create_point_ts(ta.size(), ta.start(), ta.delta(),
                                                      DoubleVector_FromNdArray(d.flatten()), 0)
            time_series[key] = array([[construct(data[fslice + (i, j)]) for j in xrange(J)]
                                     for i in xrange(I)])
        return time_series, non_time_series

    def add_time_series(self, other):
        """
        Add the time_series from the other repository to this. 
        """
        eps = 1.0e-10
        if norm(other.xx.ravel() - self.xx.ravel()) > eps or \
           norm(other.yy.ravel() - self.yy.ravel()) > eps:
            raise AromeDataRepositoryException()
        self.time_series.update(other.time_series)

    def get_sources(self, keys=None):
        """
        Convert timeseries and geo locations, to corresponding input sources, and return these as 
        shyft source vectors.
        """
        if "geo_points" not in self.other_data:
            self.other_data["geo_points"] = self._construct_geo_points()
        pts = self.other_data["geo_points"]
        if keys is None:
            keys = self.time_series.keys()
        elif isinstance(keys, str):
            keys = [keys]
        time_series = self.time_series
        sources = {}
        all_ = slice(None)
        for key in keys:
            ts = self.time_series[key]
            if key not in self.source_type_map:
                continue
            tpe = self.source_type_map[key]
            sources[key] = tpe.vector_t([tpe(GeoPoint(*pts[idx + (all_,)]), ts[idx]) for idx in 
                            ndindex(pts.shape[:-1])])
        return sources 
