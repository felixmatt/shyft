"""
Module description:
Should contain the abstract base-classes for the repositories in enki.
Brief intro to Repository pattern here:
http://martinfowler.com/eaaCatalog/repository.html
http://www.codeproject.com/Articles/526874/Repositorypluspattern-cplusdoneplusright
https://msdn.microsoft.com/en-us/library/ff649690.aspx
http://www.remondo.net/repository-pattern-example-csharp/

According to architecture diagram/current code we do have
repositories for

* configuration - helps *orchestation* to assemble data (region, model, sources etc) and repository impl.
* timeseries  - for input observations,forecasts, runoff timeseries, as well as storing scalar results
                to consider: Most of these are attributes to the region/model configuration objects, so in that
                sense, we could consider a slighty more high-level type of repositories for these.
* cells       - for reading distributed static cell information(typicall GIS system) for a given region/model spec.
* state       - for reading model-state, cell-level (snapshot of internal statevariables of the models).

each repository should provide/take enki-type classes, so that the orchestration

"""

from abc import ABCMeta, abstractmethod
import warnings


class BaseRepository(object):
    """ This base class resembles the functionality we need in dealing with states """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self, key):
        """Retrieve entry identified by key."""
        pass

    @abstractmethod
    def put(self, key, entry):
        """Write entry to repository with key as identifier."""
        pass

    @abstractmethod
    def delete(self, key):
        """Delete entry with key in repository."""
        pass

    @abstractmethod
    def find(self, *args, **kwargs):
        """Find entries in the repository matching condition and tags and return a list of keys for these entries."""
        pass


class DictRepository(BaseRepository):
    """Simple repository that uses a dictionary to hold the data."""

    def __init__(self):
        self.data = {}

    def get(self, key):
        if key in self.data:
            return self.data[key]
        raise RuntimeError("No entry with key {} found.".format(key))

    def put(self, key, entry):
        if key not in self.data:
            self.data[key] = entry
        else:
            warnings.warn("Key '%s' already exists.  Skipping without overwriting.")

    def delete(self, key):
        if key in self.data:
            return self.data.pop(key)
        raise RuntimeError("No entry with key {} found.".format(key))

    def find(self, *args, **kwargs):
        return self.data.keys()


class BaseTimeSeriesRepository(object):
    """ The abstract baseclass that defines the expectations for a TimeSeriesRepository
        Timeseries in this context is scalar timeseries, where the point is (time,value)
        Currently defined as the read and the store method.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self, ts_id_list, period):
        """read and return timeseries identified by
        Keyword arguments:
        ts_id_list -- any iterable string list, that uniquely identifies the ts in the underlying system
        period -- of type api.utcperiod specifying the wanted interval [start,end>
        """
        pass

    @abstractmethod
    def store(self, timeseries_dict):
        """ Store supplied timeseries to the underlying timeseries system.
           if the named identities does not exist, they should be created automatically as a part
           of the storage operation.

        Keyword arguments:
            timeseries_dict -- a dictionary where the key are ts identifiers, and the values are of type enki.timeseries

        """
        pass


class BaseCellRepository(object):
    """ A cell repository provides static geo properties for a geographic region like:
        {geo_position, area, (glacier,reservoir,lake,forest)_fraction, land_type, shapes}
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def number_of_cells(self):
        pass

    @abstractmethod
    def get(self, cell_property_key):
        """ return a list[number_of_cells] with the cell_property_key
        if the requested cell_property_key is not supported, an exception is raised
        """
        pass

    #
    # TODO:
    #    should we define enki.api.CellGeoInfo {catchment_id,position,area, land_..fractions.., aspects,...}
    #    concider specific methods instead?
    # @abstractmethod
    # def get_cells(self, boundingbox, catchment_ids )
    #   returning api.GeoCellVector, or at least [api.GeoCell]
    #   when creating a model cell,
    #   combine with cell-state (model & time specific)
    #                cell-model-params (model specific)
    #
