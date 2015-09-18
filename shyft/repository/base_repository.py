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


import warnings
from abc import ABCMeta, abstractmethod
from ..orchestration2.base_config import BaseSourceRepository as BaseRepository


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
