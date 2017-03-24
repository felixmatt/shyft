from __future__ import absolute_import
from builtins import range
from . import interfaces

class GeneratedStateRepositoryError(Exception):
    pass


class GeneratedStateRepository(interfaces.StateRepository):

    def __init__(self, model, init_values=None):
        self.model = model
        self.state_t = model.state_t
        self.state_vec_t = model.state_t.vector_t
        self.n = None
        self.init_values = init_values
        #self.name_map = {"kirchner": "kirchner", "gamma_snow": "gs", "skaugen_snow": "ss", "hbv_snow": "hs"}

    def find_state(self, region_model_od_criteria=None, utc_period_criteria=None,
                   tag_criteria=None):
        return interfaces.StateInfo()

    def get_state(self, state_id):
        if self.n is not None:
            state_vct = self.state_vec_t()
            state = self.state_t()
            if self.init_values is not None:  # Override default values with user-specified values
                for s_type_name, value_ in self.init_values.items():
                    #if s_type_name in self.name_map:
                    ##if hasattr(state, self.name_map[s_type_name]):
                    if hasattr(state, s_type_name):
                        #sub_state = getattr(state, self.name_map[s_type_name])
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
                    #else:
                    #    raise GeneratedStateRepositoryError("Unknown routine '{}'".format(s_type_name))
            for i in range(self.n):
                state_vct.append(state)
            return state_vct
        else:
            raise GeneratedStateRepositoryError('Number of cells should be set before getting state vector.')

    def put_state(self, region_model_id, utc_timestamp, region_model_state, tags=None):
        pass

    def delete_state(self, state_id):
        pass
