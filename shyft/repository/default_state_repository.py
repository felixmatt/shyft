from __future__ import absolute_import
from builtins import range
from . import interfaces


class DefaultStateRepository(interfaces.StateRepository):

    def __init__(self, model, n):
        self.state_t = model.state_t
        self.state_vec_t = model.state_t.vector_t
        self.n = n

    def find_state(self, region_model_od_criteria=None, utc_period_criteria=None,
                   tag_criteria=None):
        return interfaces.StateInfo()

    def get_state(self, state_id):
        v=self.state_vec_t()
        for i in range(self.n):
            v.append(self.state_t())
        return v #self.state_vec_t([self.state_t() for _ in range(self.n)]) #TODO list convertible state not yet there..

    def put_state(self, region_model_id, utc_timestamp, region_model_state, tags=None):
        pass

    def delete_state(self, state_id):
        pass
