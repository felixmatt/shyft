from abc import ABCMeta, abstractmethod

"""Module description: This module contain the abstract base-classes for the
repositories in shyft, defining the contracts that a repository should
implement.

Brief intro to Repository pattern here:
    http://martinfowler.com/eaaCatalog/repository.html
    http://www.codeproject.com/Articles/526874/Repositorypluspattern-cplusdoneplusright
    https://msdn.microsoft.com/en-us/library/ff649690.aspx
    http://www.remondo.net/repository-pattern-example-csharp/

According to architecture diagram/current code we do have
repositories for

* region-model - for reading/providing the region-model, consisting of
                 cell/catchment information, (typicall GIS system) for a given
                 region/model spec.


* state         - for reading region model-state, cell-level (snapshot of
                  internal state variables of the models).

* geo-located time-series
                - for input observations,forecasts, run-off time-series, that is
                 useful/related to the region model. E.g. precipitation,
                 temperature, radiation, wind-speed, relative humidity and even
                 measured run-off, and other time-series that can be utilized
                 by the region-model. Notice that this repository can serve
                 most type of region-models.

* configuration - helps *orchestration* to assemble data (region, model,
                  sources etc) and repository impl.

We try to design the interfaces, input types, return types, so that the number
of lines needed in the orchestration part of the code is kept to a minimum.

This implies that the input arguments to the repositories are types that goes
easily with the shyft.api. The returned types should also be shyft.api
compatible types, - thus the orchestrator can just pass on values returned into
the shyft.api.
"""


class RegionModelRepository(object):
    """
    Interface for RegionModel objects.

    The responsibility is to provide shyft.api RegionModel objects to the
    orchestrator, hiding away any implementation specific details regarding how
    the model is stored (e.g. just a mock-model, a netcdf-file based model, a
    GIS-system based model etc.).
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_region_model(self, region_id, catchments=None):
        """
        Return a fully specified shyft api region_model for region_id.

        Parameters
        -----------
        region_id: string
            unique identifier of region in data, helps
            repository to identify which model, *including model type*
            and other stuff needed in order to return a fully
            operational model.
        catchments: list of unique integers
            catchment indices when extracting a region consisting of a subset
            of the catchments

        Returns
        -------
        region_model: shyft.api type

        ```
        # Pseudo code below
        # Concrete implementation must construct cells, region_parameter
        # and catchment_parameters and return a region_model

        # Use data to create cells
        cells = [region_model.cell_type(*args, **kwargs) for cell in region]

         # Use data to create regional parameters
        region_parameter = region_model.parameter_type(*args, **kwargs)

        # Use data to override catchment parameters
        catchment_parameters = {}
        for all catchments in region:
            catchment_parameters[c] = region_model.parameter_type(*args,
                                                                  **kwargs)
        return region_model(cells, region_parameter, catchment_parameters)
        ```
        """
        pass


class StateInfo(object):
    """
    Keeps needed information for a persisted region model state.  Currently,
    the StateInfo of a region model is unique to the model, but we plan to do
    some magic so that you can extract part of state in order to transfer/apply
    to other state.

    This simple structure is utilized by the StateRepository to provide
    information about the stored state

    state_id
    A unique identifier for this state, -note that there might be infinite
    number of a state for a specific region-model/time

    region_model_id
    As a minimum the state contains the region_model_id that uniquely (within
    context) identifies the model that the state originated from.

    utc_timestamp
    The point in time where the state was sampled

    tags
    Not so important, but useful text list that we think could be useful in
    order to describe the state

    """

    def __init__(self, state_id=None, region_model_id=None,
                 utc_timestamp=None, tags=None):
        self.state_id = state_id
        self.region_model_id = region_model_id
        self.utc_timestamp = utc_timestamp
        self.tags = tags


class StateRepository(object):
    """
    Provides the needed functionality to maintain state for a region-model.

    We provide simple search functionality, based on StateInfo. The class
    should provide ready to use shyft.api type of state for a specified
    region-model. It should also be able to stash away new states for later
    use/retrieval.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def find_state(self, region_model_id_criteria=None,
                   utc_period_criteria=None, tag_criteria=None):
        """
        Find the states in the repository that matches the specified criterias
        (note: if we provide match-lambda type, then it's hard to use a db to
        do the matching)

        Parameters
        ----------
        region_model_id_criteria:
            match-lambda, or specific string, list of strings
        utc_period_criteria:
            match-lambda, or period
        tag_criteria:
            match-lambda, or list of strings ?

        Returns
        -------
        List of StateInfo objects that matches the specified criteria
        """
        pass

    @abstractmethod
    def get_state(self, state_id):
        """
        Parameters
        ----------
        state_id: string
            unique identifier of state

        Returns
        -------
        The state for a specified state_id, - the returned object/type can be
        passed directly to the region-model
        """
        pass

    @abstractmethod
    def put_state(self, region_model_id, utc_timestamp,
                  region_model_state, tags=None):
        """
        Persist the state into the repository, assigning a new unique state_id,
        so that it can later be retrieved by that return state_id assigned.
        Parameters
        ----------
        region_model_id: string
            name of the model
        utc_timestamp:utctime
            time for which the state is (considered) valid
        region_model_state:string
            something that can be interpreted as state elsewhere
        tags:list of string
            optional, tags can be associated with a state so that it can be filtered later.
            note: we are not sure if this is useful, so it's optional feature
        Returns
        -------
        state_id: immutable id
            Identifier that can be used as argument to get_state.
        """
        pass

    @abstractmethod
    def delete_state(self, state_id):
        """
        Delete the state associated with state_id,
        Throws
        ------
        StateRepositoryError: if invalid state_id, or not able to delete the
        state (access rights)

        Parameters
        ----------
        state_id: immutable id
            Identifier that uniquely identifies the state


        """
        pass


class GeoTsRepository(object):
    """
    Interface for GeoTsRepository (Geo Located Timeseries) objects.

    Responsibility:
     - to provide all hydrology relevant types of geo-located time-series,
       forecasts and ensembles needed for region-model inputs/calibrations.


    These are typical (but not limited to)
        precipitation
        temperature
        wind (speed,direction)
        radiation
        relative humidity
        snow (depth,snow water equivalent, other snow-stuff,
              like coverage etc.)
        runoff/discharge (historical observed, we use this for calibration)

    geo-located time-series def:
        A time-series where the geographic location, (area) for which the
        values apply is well defined.

        For historical observations, we usually have point observations
        (xyz + coord-system id). Forecasts, might or might not be points, they
        could be a grid-shape. So far these have been treated as points (centre
        of grid-shape).
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_timeseries(self, input_source_types, utc_period, geo_location_criteria=None):
        """
        Parameters
        ----------
        input_source_types: list
            List of source types to retrieve (precipitation,temperature..)
        utc_period: api.UtcPeriod
            The utc time period that should (as a minimum) be covered.
        geo_location_criteria: object
            Some type (to be decided), extent (bbox + coord.ref)

        Returns
        -------
        geo_loc_ts: dictionary
            dictionary keyed by ts type, where values are api vectors of geo
            located timeseries.
            Important notice: The returned time-series should at least cover the
            requested period. It could return *more* data than in
            the requested period, but must return sufficient data so
            that the f(t) can be evaluated over the requested period.
        """
        pass

    @abstractmethod
    def get_forecast(self, input_source_types, utc_period, t_c, geo_location_criteria=None):
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
        geo_loc_ts: dictionary
            dictionary keyed by ts type, where values are api vectors of geo
            located timeseries.
            Important notice: The returned forecast time-series should at least cover the
            requested period. It could return *more* data than in
            the requested period, but must return sufficient data so
            that the f(t) can be evaluated over the requested period.
        """
        pass

    @abstractmethod
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
        pass


class InterpolationParameterRepository(object):
    """
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_parameters(self, interpolation_id):
        """
        Parameters
        ----------
        interpolation_id: identifier (int| string)
            unique identifier within this repository that identifies one set of interpolation-parameters
        Returns
        -------
        parameter: shyft.api type
            Interpolation parameter object
        """
        pass


class BoundingRegion(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def bounding_box(self, epsg):
        """
        Parameters
        ----------
        epsg: string
            epsg id of the resulting coordinates

        Returns
        -------
        x: np.ndarray
           x coordinates of the four corners, numbered clockwise from
           upper left corner.
        y: np.ndarray
           x coordinates of the four corners, numbered clockwise from
           upper left corner.
        """
        pass

    @abstractmethod
    def bounding_polygon(self, epsg):
        """
        Parameters
        ----------
        epsg: string
            epsg id of the resulting coordinates

        Returns
        -------
        x: np.ndarray
           x coordinates of the smallest bounding polygon of the region
        y: np.ndarray
           y coordinates of the smallest bounding polygon of the region
        """
        pass

    @abstractmethod
    def epsg(self):
        """
        Returns
        -------
        epsg: string
            Epsg id of coordinate system
        """
        pass


class InterfaceError(Exception):
    pass


class TsStoreItem(object):
    """Represent a minimal mapping between the destination_id in the
    ts-store (like SmG, ts-name), and the lambda/function that from a
    supplied model extracts, and provides a ts possibly
    transformed/clipped to the wanted resolution and range.

    See: TimeseriesStore for usage

    """

    def __init__(self, destination_id, extract_method):
        """
        Parameters
        ----------
        destination_id: string or id
            meaningful for the time-series store, identifes a unique time-series

        extract_method: callable
            takes shyft.api model as input and return a shyft.api time-series
        """
        self.destination_id = destination_id
        self.extract_method = extract_method


class TimeseriesStore(object):
    """Represent a repository, that is capable of storing time-series
    (with almost no metadata, or they are provided elsewhere) Typical
    example would be Powel SmG, a netcdf-file-based ts-store etc.

    The usage of this class would be as a final step in orchestration,
    where we would like to save some simulation result to a database.

    """

    def __init__(self, tss, ts_item_list):
        """
        Parameters
        ----------
        tss: TsRepository
             provides a method tss.store({ts_id:shyft.api.TimeSeries})
        ts_item_list: TsStoreItem
             provide a list of mappings between ts_id and model extract function
             ref. to TsStoreItem class for description.
        """
        self.tss = tss
        self.ts_item_list = ts_item_list

    def store_ts(self, region_model, is_forecast=False):
        """
        Extracts time-series from the region_model, according to
        the ts_item_list (ref. to constructor description and TsStoreItem)
        and store the result by means of the self.tss.store method.

        Parameters
        ----------
        region_model: shyft.api model type, like PTGSKModel
            the model is passed to the extract-methods so that correct ts
            is fetched out.

        Returns
        -------
        True if storing all ts went well, otherwise False
        """
        tsid_ts_map = {tsi.destination_id: tsi.extract_method(region_model) for tsi in self.ts_item_list}
        return self.tss.store(tsid_ts_map, is_forecast)
