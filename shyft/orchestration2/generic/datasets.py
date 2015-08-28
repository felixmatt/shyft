"""
Module for reading a list of dataset sources and destinations needed for an SHyFT run.
"""

from __future__ import absolute_import

from ..base_config import BaseDatasets, BaseSourceDataset
from ..utils import get_class
from shyft import api


class Datasets(BaseDatasets):

    def __repr__(self):
        return "%s(sources=%r, destinations=%r)" % (
            self.__class__.__name__, self._source_repos,  self._destination_repos)

    def fetch_sources(self, input_source_types=None, period=None):
        """Method for fetching the data in (possibly hybrid) sources.

        This is relying in repositories specified in config files.

        Parameters
        ----------
        input_source_types : dict
            A map between the data to be extracted and the data containers in shyft.api.  Optional.
        period : tuple
            A (start_time, stop_time) tuple that species the simulation period.

        """
        if input_source_types is None:
            input_source_types = {"temperature": api.TemperatureSource,
                                  "precipitation": api.PrecipitationSource,
                                  "radiation": api.RadiationSource,
                                  "wind_speed": api.WindSpeedSource,
                                  "relative_humidity": api.RelHumSource}
        # Prepare data container to be returned
        data = {k: v.vector_t() for (k, v) in input_source_types.iteritems()}

        self._source_repos = []
        for source in self.sources:
            self._source_repos.append(source['repository'])
            repo_cls = get_class(source['class'])
            repo_instance = repo_cls(self._config_file)
            if not isinstance(repo_instance, BaseSourceDataset):
                raise ValueError("The repository class is not an instance of 'BaseSourceDataset'")
            repo_instance.fetch_sources(input_source_types, data, source['params'], period)

        return data

    def store_destinations(self):
        """Method for storing the data in (possibly hybrid) destinations.
        """
        # XXX TBD
        pass
