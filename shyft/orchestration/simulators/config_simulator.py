from .. import simulator


class ConfigSimulator(simulator.DefaultSimulator):
    def __init__(self, config):
        super().__init__(config.region_id,config.interpolation_id,config.region_model,
                         config.geo_ts, config.interp_repos)