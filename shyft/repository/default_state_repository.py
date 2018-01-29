from __future__ import absolute_import
from builtins import range
from . import interfaces


class DefaultStateRepository(interfaces.StateRepository):

    def __init__(self, model):
        self.state_t = model.state_t
        self.state_with_id_t = model.state_with_id_t
        self.state_vec_t = model.state_with_id_t.vector_t
        self.model = model

    def find_state(self, region_model_od_criteria=None, utc_period_criteria=None,
                   tag_criteria=None):
        return interfaces.StateInfo()

    def get_state(self, state_id):
        state = self.state_t()
        state_vct=self.state_vec_t()
        state_with_id = self.state_with_id_t()
        for cell in self.model.cells:
            state_with_id.state = state
            state_with_id.id = state_with_id.cell_state(cell.geo)
            state_vct.append(state_with_id)
        return state_vct #self.state_vec_t([self.state_with_id_t() for _ in self.model.cells]) #TODO list convertible state not yet there..

    def put_state(self, region_model_id, utc_timestamp, region_model_state, tags=None):
        pass

    def delete_state(self, state_id):
        pass
