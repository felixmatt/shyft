import interfaces
from .. import api


class InterpolationParameterRepository(interfaces.InterpolationParameterRepository):

    def __init__(self, model_config):
        self.params = model_config.interpolation_parameters()

    def get_parameters(self, interpolation_id):
        btk = self.params["btk"]
        btk_param = api.BTKParameter(btk["gradient"], btk["gradient_sd"], btk["sill"],
                                     btk["nugget"], btk["range"], btk["zscale"])
        idw = self.params["idw"]
        idw_args = [idw["precipitation_gradient"], idw["max_members"], idw["max_distance"]]
        prec_param = api.IDWPrecipitationParameter(*idw_args)
        del idw_args[0]
        ws_param = api.IDWParameter(*idw_args)
        rad_param = api.IDWParameter(*idw_args)
        rel_hum_param = api.IDWParameter(*idw_args)
        return api.InterpolationParameter(btk_param, prec_param, ws_param, rad_param, rel_hum_param)
