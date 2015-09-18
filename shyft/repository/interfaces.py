
from abc import ABCMeta, abstractmethod

"""
Module description:
This module contain the abstract base-classes for the repositories in shyft, defining the
contracts that a repository should implement.

Brief intro to Repository pattern here:
    http://martinfowler.com/eaaCatalog/repository.html
    http://www.codeproject.com/Articles/526874/Repositorypluspattern-cplusdoneplusright
    https://msdn.microsoft.com/en-us/library/ff649690.aspx
    http://www.remondo.net/repository-pattern-example-csharp/

According to architecture diagram/current code we do have
repositories for

* region-model - for reading/providing the region-model, consisting of cell/catchment information,
                 (typicall GIS system) for a given region/model spec.
                 
                 
* state         - for reading region model-state, cell-level (snapshot of internal state variables of the models).

* input time-series  - for input observations,forecasts, runoff timeseries, that is useful/related to the
                    region model. E.g. precipitation, temperature, radiation, wind-speed, relative humidity and
                    even measured run-off, and other time-series that can be utilized by the region-model.
                    Notice that this repository can serve most type of region-models.

* configuration - helps *orchestation* to assemble data (region, model, sources etc) and repository impl.

We try to design the interfaces, input types, return types, so that the number of lines needed in the orchestration
part of the code is keept to a minimum.

This implies that the input arguments to the repositories are types that goes easily with the shyft.api.
The returned types should also be shyft.api compatible types, - thus the orchestrator can just pass on
values returned into the shyft.api.


"""

class RegionRepository(object):
    """Interface for RegionModel  objects. 
       The responsibility is to provide shyft.api RegionModel objects to the orchestrator,
       hiding away any implementation specific details regarding how the model is stored
       (e.g. just a mock-model, a netcdf-file based model, a GIS-system based model etc.).
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_region(self, region_id, region_model, catchments=None):
        """
        Return a fully specified region_model for region_id.
        
        Parameters
        -----------
        region_id: string
            unique identifier of region in data
        region_model: shyft.api type
            model to construct. Has cell constructor and region/catchment
            parameter constructor.
        catchments: list of unique integers
            catchment indices when extracting a region consisting of a subset
            of the catchments
        has attribs to construct  params and cells etc.

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