import yaml


def save_config(yaml_filename, object_):
    with open(yaml_filename, 'w') as yfile:
        yfile.write(yaml.dump(object_, default_flow_style=False))


def load_config(yaml_filename):
    with open(yaml_filename, 'r') as yfile:
        return yaml.load(yfile)
