from __future__ import absolute_import

from .interfaces import StateInfo, StateRepository

import os
from glob import glob
import yaml
from shyft import api
from shyft.api.pt_gs_k import PTGSKStateIo


class StateRepositoryError(Exception):
    pass


class State(object):
    """
    Module internal only, persisted state in the yaml-file, kept equal to
    orchestration for backward compatibility.
    """

    def __init__(self, state_list, utc_timestamp=None, tags=None):
        self.data = {"utc_timestamp": utc_timestamp,
                     "tags": tags,
                     "state": state_list
                     }

    @property
    def utc_timestamp(self):
        return self.data["utc_timestamp"]

    @utc_timestamp.setter
    def utc_timestamp(self, new_utc_timestamp):
        self.data["utc_timestamp"] = new_utc_timestamp

    @property
    def state_list(self):
        return self.data["state"]

    @property
    def tags(self):
        return self.data["tags"]

    @tags.setter
    def tags(self, new_tags):
        self.data["tags"] = new_tags

    def __len__(self):
        return len(self.state_list)


class YamlStateRepository(StateRepository):
    """
    Local yaml-file storage of states.

    The states are yaml-files, matching a certain filename-pattern
    each file contains one region model state.
    e.g.: name-convention could be neanidelv-ptgsk_20130101T0101_version.yaml
    the unique id is the filename
    the version number is not guaranteed to be increasing
    .. we could utilize file create/modified time
    """

    def __init__(self, directory_path, state_serializer=None, file_pattern="*.yaml"):
        """
        Parameters
        ----------
        directory_path: string
            should point to the directory where the state files are stored
        state_serializer: object
            Instance of a class capable of converting a
            shyft.api.<methodstack>statevector to string and vice-versa
        file_pattern: string
            Glob pattern like '*.yaml'
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        self._directory_path = directory_path
        self._sio = state_serializer if state_serializer is not None else PTGSKStateIo()
        self._file_pattern = file_pattern
        self._filename_item_separator = "@"  # We encode the filename with info separated by this

    def _get_state_file_list(self, state_directory, glob_pattern):
        full_path = ""
        if os.path.isabs(state_directory):
            full_path = state_directory
        else:
            cwd = os.getcwd()
            if os.path.isdir(os.path.join(cwd, state_directory)):
                full_path = os.path.realpath(os.path.join(cwd, state_directory))
            elif os.path.isdir(os.path.join(os.path.realpath(__file__), state_directory)):
                full_path = os.path.realpath(
                    os.path.join(os.path.realpath(__file__, state_directory)))
            else:
                raise StateRepositoryError("Can not find directory {}, please check your configuration setup.".format(state_directory))

        if glob_pattern is None:
            glob_pattern = "*.yaml"

        return glob(os.path.join(full_path, glob_pattern))

    def _save_state_as_yaml_file(self, state, filename, **params):
        with open(filename, "w") as f:
            f.write(yaml.dump(state, **params))

    def _load_state_from_yaml_file(self, filename):
        with open(filename, "r") as f:
            contents = yaml.load(f.read())
            return contents

    def _unique_id(self, region_model_id, utc_timestamp):
        """
        Given region_model_id and utc_timestamp, return unique id
        (filename within directory) notice that we encode part of the
        state info into the filename, and utilizes the directory as
        the container (ensuring unique names).

        Returns
        -------
        uid: string
            unique id
        """
        if self._filename_item_separator in region_model_id:
            raise StateRepositoryError("{} is illegal character in region_model_id ".format(self._filename_item_separator))
        version = 0
        state_id = ""
        # ms does not tolerate : in filenames
        utc_timestamp_str = api.Calendar().to_string(utc_timestamp).replace(":", ".")
        while True:
            version += 1
            state_id = "{}{}{}{}{}.yaml".format(region_model_id,
                                                self._filename_item_separator,
                                                utc_timestamp_str,
                                                self._filename_item_separator,
                                                version)
            if not os.path.exists(os.path.join(self._directory_path, state_id)):
                break
        return state_id

    def _state_info_from_filename(self, filename):
        """
        Expect filename in format:
        <region-model-id>_<YYYY.MM.DDThh:mm:ss>_<version>.yaml

        Returns
        -------
        state_info: StateInfo
            Containing all except tags filled in..(limitation currently..)
            or None if the filename did not conform to standard.
        """
        try:
            parts = filename.split(self._filename_item_separator)
            si = StateInfo()
            si.region_model_id = parts[0]
            si.state_id = filename
            ymd = parts[1].split("T")[0].split(".")  # TODO: can datetime parse the format directly?
            hms = parts[1].split("T")[1].split(".")
            utc_calendar = api.Calendar()
            si.utc_timestamp = utc_calendar.time(api.YMDhms(int(ymd[0]), int(ymd[1]), int(ymd[2]),
                                                            int(hms[0]), int(hms[1]), int(hms[2])))
            return si
        except:
            return None

    def find_state(self, region_model_id_criteria=None,
                   utc_timestamp_criteria=None, tag_criteria=None):
        """
        Find the states in the repository that matches the specified
        criterias (note: if we provide match-lambda type, then it's
        hard to use a db to do the matching).

        Parameters
        ----------
        region_model_id_criteria: string
            Exact match string (could be extended).
        utc_timestamp_criteria:  long
            Utc timestamp: Requires region_model_id criteria as well.
            If spec. return state with highest possible time before critera.
        tag_criteria: callable or list of strings
            Match-lambda, or list of strings?

        Returns
        -------
            state_infos: list
                List of StateInfo objects that match the specified criteria.
        """
        if utc_timestamp_criteria is not None and region_model_id_criteria is None:
            raise StateRepositoryError("You have to specify a region_model_id AND a utc_timestamp_critera")

        file_list = self._get_state_file_list(self._directory_path, self._file_pattern)
        res = []
        for filename in file_list:
            si = self._state_info_from_filename(os.path.basename(filename))
            if si is not None:
                if region_model_id_criteria is None or \
                   si.region_model_id == region_model_id_criteria:
                    res.append(si)
            # NOTE: in this implementation you have to specify regon_model_id AND time_spec..
        # Find state with max.utc_timestamp <= criteria
        if len(res) > 0 and utc_timestamp_criteria is not None:
            item = None
            for e in res:
                if e.utc_timestamp <= utc_timestamp_criteria:
                    if item is None or (e.utc_timestamp > item.utc_timestamp):
                        item = e
            if item is None:
                res = []
            else:
                res = [item]
        return res

    def get_state(self, state_id):
        """
        Return
        ------
        state: shyft.api object
            The state for a specified state_id, -the returned object/type can be
            passed directly to the region-model.
        """
        state = self._load_state_from_yaml_file(os.path.join(self._directory_path, state_id))
        return self._sio.vector_from_string(state.state_list)

    def put_state(self, region_model_id, utc_timestamp, region_model_state, tags=None):
        """
        Persist the state into the repository,
        assigning a new unique state_id, so that it can later be retrieved by that

        Returns
        -------
        state_id: string
            Unique id assigned to state
        """
        state_id = self._unique_id(region_model_id, utc_timestamp)
        # TODO: save state here
        persisted_form = State(self._sio.to_string(region_model_state), utc_timestamp, tags)
        self._save_state_as_yaml_file(persisted_form, os.path.join(self._directory_path, state_id))
        return state_id

    def delete_state(self, state_id):
        """ simply delete the state from the repository (removing the file..) """
        full_file_path = os.path.join(self._directory_path, state_id)
        if os.path.exists(full_file_path):
            os.unlink(full_file_path)
