from shyft import pt_gs_k
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
    sio = pt_gs_k.PTGSKStateIo()
    return sio.vector_from_string(data)


def extract_ptgsk_model_state_as_string(model):
    sio = pt_gs_k.PTGSKStateIo()
    state_vector = pt_gs_k.PTGSKStateVector()
    model.get_states(state_vector)
    return sio.to_string(state_vector)


def extract_ptgsk_model_state(model):
    return State(extract_ptgsk_model_state_as_string(model))


def build_ptgsk_model_state_from_data(data):
    return build_ptgsk_model_state_from_string(data)


def set_ptgsk_model_state(model, state):
    state_vector = build_ptgsk_model_state_from_data(state.state_list)
    if len(state_vector) != model.size():
        raise RuntimeError("The size of the model does not coinside with \
the length of the state vector")
    model.set_states(state_vector)


def save_state_as_yaml_file(state, filename):
    with open(filename, "w") as of:
        of.write(yaml.dump(state, default_flow_style=False))


def load_state_from_yaml_string(string):
    return yaml.load(string)
