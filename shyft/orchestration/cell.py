class Cell(object):
    """Mother of all cells structure, probably will not be useful."""

    def __init__(self, **kwargs):
        self.data = {"catchment_id": None,
                     "land_type": None,
                     "geo_position": None,
                     "glacier_fraction": None,
                     "initial_bare_ground_fraction": None,
                     "snow_cv": None,
                     "radiation_slope_factor": None,
                     "area": None}
        self.data.update(kwargs)

    @property
    def catchment_id(self):
        return self.data["catchment_id"]

    @property
    def land_type(self):
        return self.data["land_type"]

    @property
    def geo_position(self):
        return self.data["geo_position"]

    @property
    def glacier_fraction(self):
        return self.data["glacier_fraction"]

    @property
    def initial_bare_ground_fraction(self):
        return self.data["initial_bare_ground_fraction"]

    @property
    def snow_cv(self):
        return self.data["snow_cv"]

    @property
    def radiation_slope_fraction(self):
        return self.data["radiation_slope_fraction"]

    @property
    def area(self):
        return self.data["area"]

