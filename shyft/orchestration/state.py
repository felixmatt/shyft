from shyft import api
import yaml

class State(object):

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

def build_ptgsk_model_state_from_string(data):
    sio=api.PTGSKStateIo()
    return sio.vector_from_string(data)

def extract_ptgsk_model_state_as_string(model):
    sio=api.PTGSKStateIo();
    state_vector=api.PTGSKStateVector()
    model.get_states(state_vector)
    return sio.to_string(state_vector)

def extract_ptgsk_model_state(model):
    return State(extract_ptgsk_model_state_as_string(model))
    #state_list = []
    #states = api.PTGSKStateVector() # Need return by reference here due to a difficult swig issue I can't resolve.
    #model.get_end_states(states)
    #for i in xrange(len(states)):
    #    state = states[i]
    #    state_list.append({"pt": {},
    #                   "gs": {"albedo": state.gs.albedo,
    #                          "lwc": state.gs.lwc,
    #                          "surface_heat": state.gs.surface_heat,
    #                          "alpha": state.gs.alpha,
    #                          "sdc_melt_mean": state.gs.sdc_melt_mean,
    #                          "acc_melt": state.gs.acc_melt,
    #                          "iso_pot_energy": state.gs.iso_pot_energy,
    #                          "temp_swe": state.gs.temp_swe
    #                         },
    #                   "kirchner": {"q": state.kirchner.q}
    #                  })
    #return State(state_list)

def build_ptgsk_model_state_from_data(data):
    return build_ptgsk_model_state_from_string(data)
    #state_vec = api.PTGSKStateVector()
    #for s in data:
    #    api_state = api.PTGSKStat(api.PriestleyTaylorState(**s["pt"]),
    #                              api.GammaSnowState(**s["gs"]),
    #                              api.KirchnerState(**s["kirchner"]))
    #    state_vec.append(api_state)
    #return state_vec



def set_ptgsk_model_state(model, state):
    state_vector=build_ptgsk_model_state_from_data(state.state_list)
    if len(state_vector) != model.size():
        raise RuntimeError("The size of the model does not coinside with the length of the state vector")
    model.set_states(state_vector)

def save_state_as_yaml_file(state, filename):
    with open(filename, "w") as of:
        of.write(yaml.dump(state, default_flow_style=False))
        #of.write(state.state_list)

def load_state_from_yaml_string(string):
    return yaml.load(string)
