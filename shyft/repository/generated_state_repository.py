from __future__ import absolute_import
from builtins import range
from . import interfaces

class GeneratedStateRepositoryError(Exception):
    pass


class GeneratedStateRepository(interfaces.StateRepository):

    def __init__(self, model, init_values=None):
        self.state_t = model.state_t
        self.state_vec_t = model.state_t.vector_t
        self.n = None
        self.init_values = init_values

    def find_state(self, region_model_od_criteria=None, utc_period_criteria=None,
                   tag_criteria=None):
        return interfaces.StateInfo()

    def get_state(self, state_id):
        if self.n is not None:
            if self.init_values is None:
                v=self.state_vec_t()
                for i in range(self.n):
                    v.append(self.state_t())
                return v #self.state_vec_t([self.state_t() for _ in range(self.n)]) #TODO list convertible state not yet there..
            else:
                raise GeneratedStateRepositoryError("Using user-specified init_values not supported yet. "
                                                    "Set init_values to null to use default values.")
        else:
            raise GeneratedStateRepositoryError('Number of cells should be set before getting state vector.')

    def put_state(self, region_model_id, utc_timestamp, region_model_state, tags=None):
        pass

    def delete_state(self, state_id):
        pass
