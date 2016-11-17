from . import interfaces

class GridppDataRepositoryError(Exception):
    pass


class GridppDataRepository(interfaces.GeoTsRepository):

    def __init__(self, model_data_repo, reference_data_repo):
        pass

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

        pass

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