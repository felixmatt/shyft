from __future__ import absolute_import
from builtins import range
from . import interfaces

class GeneratedStateRepositoryError(Exception):
    pass


class GeneratedStateRepository(interfaces.StateRepository):

    def __init__(self, model, init_values=None):
        self.model = model
        self.state_t = model.state_t
        self.state_with_id_t = model.state_with_id_t
        self.state_vec_t = model.state_with_id_t.vector_t
        self.init_values = init_values

    def find_state(self, region_model_od_criteria=None, utc_period_criteria=None,
                   tag_criteria=None):
        return interfaces.StateInfo()

    def get_state(self, state_id):
        state_vct = self.state_vec_t()
        state = self.state_t()
        if self.init_values is not None:  # Override default values with user-specified values
            for s_type_name, value_ in self.init_values.items():
                if hasattr(state, s_type_name):
                    sub_state = getattr(state, s_type_name)
                    for s, v in value_.items():
                        if hasattr(sub_state, s):
                            setattr(sub_state, s, v)
                        else:
                            raise GeneratedStateRepositoryError(
                                "Invalid state '{}' for routine '{}'".format(s, s_type_name))
                else:
                    raise GeneratedStateRepositoryError(
                        "Invalid routine '{}' for selected model '{}'".format(s_type_name, self.model.__name__))
        state_with_id = self.state_with_id_t()
        for cell in self.model.cells:
            state_with_id.state = state
            state_with_id.id = state_with_id.cell_state(cell.geo)
            state_vct.append(state_with_id)
        return state_vct

    def put_state(self, region_model_id, utc_timestamp, region_model_state, tags=None):
        pass

    def delete_state(self, state_id):
        pass
