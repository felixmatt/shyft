"""
Module for reading region data files needed for an SHyFT run.
"""

from __future__ import absolute_import

from ..base_config import BaseRegion
from ..utils import get_class


# This class will delegate in class section of the repository; e.g.:
# ----
# repository:
#   class: shyft.orchestration2.reference.Region
# ...
class Region(BaseRegion):

    def __init__(self, config_file):
        super(Region, self).__init__(config_file)
        # Get an instance of the class for the repository
        repo = self.repository
        repo_cls = get_class(repo['class'])
        print("data_file 0:", repo['data_file'])
        data_file = self.absdir(repo['data_file'])
        print("data_file 1:", data_file)
        repo_instance = repo_cls(config_file, data_file)
        if not isinstance(repo_instance, BaseRegion):
            raise ValueError("The repository class is not an instance of 'BaseRegion'")
        self.repo_instance = repo_instance

    def fetch_cell_properties(self, varname):
        return self.repo_instance.fetch_cell_properties(varname)

    def fetch_cell_centers(self):
        return self.repo_instance.fetch_cell_centers()

    def fetch_cell_areas(self):
        return self.repo_instance.fetch_cell_areas()

    def fetch_catchments(self, what):
        return self.repo_instance.fetch_catchments(what)
