from shyft import api

def cell_argument_factory(cell_type, *args):
        type_to_builder = {#api.FullPTHSKModel: PTHSKArgBuilder,
                           #api.ReducedPTHSKModel: PTHSKArgBuilder, 
                           api.PTGSKModel: PTGSKArgBuilder,
                           api.PTGSKOptModel: PTGSKArgBuilder}
        assert cell_type in type_to_builder
        return type_to_builder[cell_type](*args)

class PTHSKArgBuilder(object):

    def __init__(self, geo_points, lake_fractions, reservoir_fractions, forest_fractions, areas):
        self.geo_points = geo_points
        self.lake_fractions = lake_fractions
        self.reservoir_fractions = reservoir_fractions
        self.forest_fractions = forest_fractions
        self.areas = areas

    def update(self, mapped_index, arg_dict):
        self.mapped_index = mapped_index
        self.arg_dict = arg_dict

    def __getitem__(self, i):
        return [api.GeoPoint(*self.geo_points[i]), self.mapped_index, self.arg_dict["hbv_snow"]["snow_redistribution_factors"],
                self.arg_dict["hbv_snow"]["snow_quantiles"], self.lake_fractions[i], self.reservoir_fractions[i],
                self.forest_fractions[i], self.arg_dict["gamma_snow"]["initial_bare_ground_fraction"], 
                self.arg_dict["gamma_snow"]["snow_cv"], self.arg_dict["priestley_taylor"]["alpha"], 
                self.arg_dict["kirchner"]["c1"], self.arg_dict["kirchner"]["c2"], 
                self.arg_dict["kirchner"]["c3"], self.areas[i]]

class PTGSKArgBuilder(object):

    def __init__(self, geo_points, glacier_fractions, lake_fractions, reservoir_fractions, forest_fractions, areas):
        self.geo_points = geo_points
        self.glacier_fractions = glacier_fractions
        self.lake_fractions = lake_fractions
        self.reservoir_fractions = reservoir_fractions
        self.forest_fractions = forest_fractions
        self.areas = areas

    def update(self, mapped_index, arg_dict):
        self.mapped_index = mapped_index
        self.arg_dict = arg_dict

    def __getitem__(self, i):
        return [api.GeoPoint(*self.geo_points[i]), self.mapped_index, self.glacier_fractions[i], self.lake_fractions[i], self.reservoir_fractions[i],
                self.forest_fractions[i], self.arg_dict["gamma_snow"]["initial_bare_ground_fraction"], 
                self.arg_dict["gamma_snow"]["snow_cv"], self.arg_dict["priestley_taylor"]["alpha"], 
                self.arg_dict["kirchner"]["c1"], self.arg_dict["kirchner"]["c2"], 
                self.arg_dict["kirchner"]["c3"], self.areas[i]]
