from base_repository import DictRepository
from datetime import datetime
from shyft.orchestration.state import State
from shyft.orchestration.state import load_state_from_yaml_string
import os
from glob import glob

class StateRepositoryError(Exception): pass


def combine_conditions(c1, c2):
    """Simple combiner of lazy evaluation of boolean expressions.""" 
    return lambda x: c1(x) and c2(x)


class TimeCondition(object):
    """Overload comparision operators to return a callable that can be used for filtering during state entry searches."""

    def __lt__(self, utc_timestamp):
        return lambda x: x.utc_timestamp < utc_timestamp

    def __gt__(self, utc_timestamp):
        return lambda x: x.utc_timestamp > utc_timestamp

    def __le__(self, utc_timestamp):
        return lambda x: x.utc_timestamp <= utc_timestamp

    def __ge__(self, utc_timestamp):
        return lambda x: x.utc_timestamp >= utc_timestamp


class LocalStateRepository(DictRepository):
    """Local file storage of states."""
    pass


def yaml_file_storage_factory(info, state_directory, glob_pattern=None):
    # Find the location of the state_directory
    if os.path.isabs(state_directory):
        full_path = state_directory
    else:
        cwd = os.getcwd()
        if os.path.isdir(os.path.join(cwd, state_directory)):
            full_path = os.path.realpath(os.path.join(cwd, state_directory))
        elif os.path.isdir(os.path.join(os.path.realpath(__file__), state_directory)):
            full_path = os.path.realpath(os.path.join(os.path.realpath(__file__, state_directory)))
        else:
            raise StateRepositoryError("Can not find directory {}, please check your configuration setup.".format(state_directory))
        
    if glob_pattern is None:
        glob_pattern = "*.yaml"

    state_file_list = glob(os.path.join(full_path, glob_pattern))
    if not state_file_list:
        raise StateRepositoryError("No state files matching {} found in directory {}, please check your configuration setup.".format(glob_pattern, state_directory))
    states = []
    repository = LocalStateRepository()
    for state_file in state_file_list:
        with open(state_file, "r") as ff:
            print 'Loading state from yaml file...'
            repository.put(state_file, load_state_from_yaml_string(ff.read()))
            print 'Done loading state'
    return repository
