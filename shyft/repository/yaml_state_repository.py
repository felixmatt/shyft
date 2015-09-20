#from __future__ import print_function
from __future__ import absolute_import

from .interfaces import StateInfo, StateRepository

#from shyft.orchestration.state import load_state_from_yaml_string
import os
from glob import glob
import yaml
from shyft.api import Calendar,UtcPeriod,YMDhms
from shyft.api.pt_gs_k import PTGSKStateIo


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

def save_state_as_yaml_file(state, filename, **params):
    #print("write to yaml state filename:", filename)
    with open(filename, "w") as f:
        f.write(yaml.dump(state, **params))

def load_state_from_yaml_file(filename):
    #print("read from yaml state filename:", filename)
    with open(filename, "r") as f:
        contents = yaml.load(f.read())
        return contents

def get_state_file_list(state_directory,glob_pattern):
    full_path=""
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

    return glob(os.path.join(full_path, glob_pattern))

class State(object):
    """ persisted state in the yaml-file """

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
    """Local yaml-file storage of states.
    The states are yaml-files, matching a certain filename-pattern
    each file contains one region model state.
    e.g.: name-convention could be neanidelv-ptgsk_20130101T0101_version.yaml
    the unique id is the filename 
    the version number is not guaranteed to be increasing
    .. we could utilize file create/modified time
    """
    def __init__(self,directory_path,state_serializer=PTGSKStateIo(),file_pattern="*.yaml"):
        """
        Parameters:
        directory_path should point to the directory where the state files are stored
        state_serializer should be instance of a class capable of converting a shyft.api.<methodstack>statevector to string and vice-versa
        file_pattern should be *.yaml 
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        self._directory_path=directory_path
        self._sio=state_serializer
        self._file_pattern=file_pattern

    def _unique_id(self,region_model_id,utc_timestamp):
        """ given region_model_id and utc_timestamp, return unique id (filename within directory) 
            notice that we encode part of the state info into the filename, and utilizes the
            directory as the container (ensuring unique names)
        """
        if '_' in region_model_id:
            raise StateRepositoryError("'_' is illegal character in region_model_id ")
        version=0
        state_id=""
        while True:
            version =version +1
            state_id= "{}_{}_{}.yaml".format(region_model_id,Calendar().to_string(utc_timestamp),version)
            if not os.path.exists(os.path.join(self._directory_path,state_id)):
                break
        return state_id
    
    def _state_info_from_filename(self,filename):
        """ expect filename in format <region-model-id>_<YYYY.MM.DDThh:mm:ss>_<version>.yaml 
            return:
            StateInfo with all except tags filled in..(limitation currently..)
            or None if the filename did not conform to standard
        """
        try:
            parts=filename.split("_")
            si=StateInfo()
            si.region_model_id= parts[0]
            si.state_id=filename
            ymd=parts[1].split("T")[0].split(".")#TODO: does datetime parse the format directly?
            hms=parts[1].split("T")[1].split(":")
            utc_calendar=Calendar()
            si.utc_timestamp=utc_calendar.time(YMDhms(int(ymd[0]),int(ymd[1]),int(ymd[2]),int(hms[0]),int(hms[1]),int(hms[2])))
            return si
        except:
            return None
                                           
        
    def find_state(self,region_model_id_criteria=None,utc_period_criteria=None,tag_criteria=None):
        """ Find the states in the repository that matches the specified criterias
         (note: if we provide match-lambda type, then it's hard to use a db to do the matching)
         
        Parameters
        region_model_id_criteria: match-lambda, or specific string, list of strings 
        utc_period_criteria: match-lambda, or period
        tag_criteria: match-lambda, or list of strings ?
         
        return list of StateInfo objects that matches the specified criteria
        """
         
        file_list=get_state_file_list(self._directory_path,self._file_pattern)
        res=[]
        for filename in file_list:
            si= self._state_info_from_filename(os.path.basename(filename))
            if si is not None:
                res.append(si)
        #TODO add criteria matching
        return res

    def get_state(self, state_id):
        """
        return the state for a specified state_id, - the returned object/type can be passed directly to the region-model 
        """
        state= load_state_from_yaml_file(os.path.join(self._directory_path,state_id))
        return self._sio.vector_from_string(state.state_list)
    
    def put_state(self, region_model_id,utc_timestamp,region_model_state,tags=None):
        """ 
        persist the state into the repository,
        assigning a new unique state_id, so that it can later be retrieved by that
        return state_id assigned
        """
        state_id=self._unique_id(region_model_id,utc_timestamp)
        #TODO: save state here
        persisted_form= State(self._sio.to_string(region_model_state),utc_timestamp,tags)
        save_state_as_yaml_file(persisted_form,os.path.join(self._directory_path,state_id))
        return state_id
    
    def delete_state(self, state_id):
        """ simply delete the state from the repository (removing the file..) """
        full_file_path=os.path.join(self._directory_path,state_id)
        if os.path.exists(full_file_path):
            os.unlink(full_file_path)

