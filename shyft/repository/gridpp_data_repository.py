from . import interfaces
from shyft.api import BTKParameter
from shyft.api import bayesian_kriging_temperature
from shyft.api import IDWTemperatureParameter
from shyft.api import idw_temperature
from shyft.api import GeoPointVector


class GridppDataRepositoryError(Exception):
    pass


class GridppDataRepository(interfaces.GeoTsRepository):

    def __init__(self, simulated_data_repo, observed_data_repo):
        self.sim_repo = simulated_data_repo
        self.obs_repo = observed_data_repo
        self.interp_params = {}
        # kriging parameters
        self.btk_params = BTKParameter()  # we could tune parameters here if needed
        # idw parameters,somewhat adapted to the fact that we
        #  know we interpolate from a grid, with a lot of neigbours around
        self.idw_params = IDWTemperatureParameter()  # here we could tune the paramete if needed
        self.idw_params.max_distance = 20 * 1000.0  # max at 10 km because we search for max-gradients
        self.idw_params.max_members = 20  # for grid, this include all possible close neighbors
        self.idw_params.gradient_by_equation = True  # resolve horisontal component out

    def get_timeseries_sim_grid(self, input_source_types, utc_period, geo_location_criteria=None):
        return self.sim_repo.get_timeseries(input_source_types, utc_period, geo_location_criteria=geo_location_criteria)

    def get_timeseries_obs(self, input_source_types, utc_period, geo_location_criteria=None):
        return self.obs_repo.get_timeseries(input_source_types, utc_period, geo_location_criteria=geo_location_criteria)

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
        # Get the geolocated timeseries from obs_repo
        obs = self.get_timeseries_obs(input_source_types, utc_period, geo_location_criteria=geo_location_criteria)
        # Get the geolocated timeseries from sim_repo
        sim = self.get_timeseries_sim_grid(input_source_types, utc_period, geo_location_criteria=geo_location_criteria)
        # Processing for temperature variable
        sim_at_obs = {}
        sim_temp_grid = sim['temperature']
        # Create a GeoPointVector with the locations of the temperature observation points
        obs_locations_temp = GeoPointVector()
        [obs_locations_temp.append(src.mid_point()) for src in obs['temperature']]
        # Project the sim timeseries grid to the obs timseries locations
        # -> we use the same time axis as the sim
        ta = sim_temp_grid[0].ts.time_axis
        sim_at_obs['temperature'] = idw_temperature(sim_temp_grid, obs_locations_temp, ta, self.interp_params)
        return sim_at_obs

    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria=None):
        """
        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve. Valid types are:
                * relative_humidity
                * temperature
                * precipitation
                * radiation
                * wind_speed
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.
        t_c: long
            Forecast specification; return newest forecast older than t_c.
        geo_location_criteria: object
            Some type (to be decided), extent (bbox + coord.ref).

        Returns
        -------
        geo_loc_ts: dictionary
            dictionary keyed by ts type, where values are api vectors of geo
            located timeseries.
        """

        pass